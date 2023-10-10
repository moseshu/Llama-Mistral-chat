import argparse
import json
import os
import sys
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.state import PartialState
from datasets import Dataset, load_dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,set_peft_model_state_dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from trl import SFTTrainer


from my_logging import custom_logger as logger
from callbacks import LoadBestPeftModelCallback, SavePeftModelCallback
from params import LLMTrainingParams
from utils import process_data, get_target_modules
import utils
from datasets import load_dataset,Dataset,concatenate_datasets

def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()



def load_datasets(data_path):
    files = os.listdir(data_path)
    data_paths = [os.path.join(data_path,f) for f in files]
    all_datasets = []
    for file in data_paths:
        if not (file.endswith(".json") or file.endswith(".jsonl")):
            continue
        try:
            logger.info(f"load file:{file}")
            raw_dataset = load_dataset("json", data_files=file,cache_dir=".cache")
            cmd = f'rm -r .cache/'
            all_datasets.append(raw_dataset['train'])
            # subprocess.check_call(cmd, shell=True)
        except:
            print(f'load {file}file error check your file is correct json file')
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets
    

def train(config):
    if isinstance(config, dict):
        config = LLMTrainingParams(**config)

    # TODO: remove when SFT is fixed
    # if config.trainer == "sft":
    #     config.trainer = "default"

    # check if config.train_split.csv exists in config.data_path
    

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
    )

    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = config.model_max_length

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0

    # if config.trainer == "default":
    #     train_data = process_data(
    #         data=train_data,
    #         tokenizer=tokenizer,
    #         config=config,
    #     )
    #     if config.valid_split is not None:
    #         valid_data = process_data(
    #             data=valid_data,
    #             tokenizer=tokenizer,
    #             config=config,
    #         )

    model_config = AutoConfig.from_pretrained(
        config.model,
    )

    if config.use_peft:
        if config.use_int4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config.use_int4,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            config.fp16 = True
        elif config.use_int8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=config.use_int8)
            config.fp16 = True
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            config=model_config,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map={"": Accelerator().process_index} if torch.cuda.is_available() else None,
            
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            config=model_config,
            token=config.token,
            # trust_remote_code=True,
            # use_flash_attention_2=config.use_flash_attention_2,
        )

    model.resize_token_embeddings(len(tokenizer))
    if config.use_flash_attention_2:
        logger.info("use flash_attention_2")
        from training.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
        
    if config.use_peft:
        if config.use_int8 or config.use_int4:
            model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=get_target_modules(config),
        )
        model = get_peft_model(model, peft_config)
        
    if config.peft_path:
        logger.info("load adapter weights")
        adapters_weights = torch.load(f"{config.peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)

    
    model.print_trainable_parameters()
    
    if config.block_size == -1:
        config.block_size = None

    if config.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if config.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({config.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(config.block_size, tokenizer.model_max_length)

    config.block_size = block_size


    train_data = load_datasets(config.data_path)
    train_data = process_data(train_data,tokenizer,config)
    if config.trainer == "default":
        tokenize_fn = partial(utils.tokenize, tokenizer=tokenizer, config=config)
        group_texts_fn = partial(utils.group_texts, config=config)

        train_data = train_data.map(
            tokenize_fn,
            batched=True,
            num_proc=1,
            remove_columns=list(train_data.features),
            desc="Running tokenizer on train dataset",
        )

        if config.valid_split is not None:
            valid_data = valid_data.map(
                tokenize_fn,
                batched=True,
                num_proc=1,
                remove_columns=list(valid_data.features),
                desc="Running tokenizer on validation dataset",
            )

        train_data = train_data.map(
            group_texts_fn,
            batched=True,
            num_proc=4,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        if config.valid_split is not None:
            valid_data = valid_data.map(
                group_texts_fn,
                batched=True,
                num_proc=4,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    logger.info("creating trainer")
    # trainer specific
    if config.logging_steps == -1:
        if config.valid_split is not None:
            logging_steps = int(0.2 * len(valid_data) / config.batch_size)
        else:
            logging_steps = int(0.2 * len(train_data) / config.batch_size)
        if logging_steps == 0:
            logging_steps = 1

    else:
        logging_steps = config.logging_steps

    training_args = dict(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        evaluation_strategy=config.evaluation_strategy if config.valid_split is not None else "no",
        logging_steps=logging_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        gradient_accumulation_steps=config.gradient_accumulation,
        report_to="tensorboard",
        auto_find_batch_size=config.auto_find_batch_size,
        lr_scheduler_type=config.scheduler,
        optim=config.optimizer,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        load_best_model_at_end=True if config.valid_split is not None else False,
        ddp_find_unused_parameters=False,
    )

    args = TrainingArguments(**training_args)

    callbacks = []
    if config.use_peft:
        callbacks.append(SavePeftModelCallback)
        if config.valid_split is not None:
            callbacks.append(LoadBestPeftModelCallback)

    trainer_args = dict(
        args=args,
        model=model,
    )

    if config.trainer == "default":
        logger.info(f"""if you want to use SFTTrainer,please set trainer=="sft" """)
        trainer = Trainer(
            **trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            callbacks=callbacks,
        )
    elif config.trainer == "sft":
        logger.info(f"""if you want to use transformer.Trainer,please set trainer=="default" """)
        trainer = SFTTrainer(
            **trainer_args,
            train_dataset=train_data,
            eval_dataset=valid_data if config.valid_split is not None else None,
            peft_config=peft_config if config.use_peft else None,
            dataset_text_field=config.text_column,
            max_seq_length=config.block_size,
            tokenizer=tokenizer,
            packing=True,
        )
    else:
        raise ValueError(f"trainer `{config.trainer}` not supported,set trainer='sft' or 'default'")
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    for name, module in trainer.model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if script_args.bf16:
        #         module = module.to(torch.bfloat16)
        if "norm" in name and not config.use_flash_attention_2:
            module = module.to(torch.float32)
        # if "lm_head" in name or "embed_tokens" in name:
        #     if hasattr(module, "weight"):
        #         if script_args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)

    trainer.train()

    logger.info("Finished training, saving model...")
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    args = parse_args()
    training_config = json.load(open(args.training_config))
    config = LLMTrainingParams(**training_config)
    logger.info(config)
    train(config)
