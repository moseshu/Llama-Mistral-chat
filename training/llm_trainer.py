import os
import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset,Dataset
import json
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from itertools import chain
from datasets import Dataset, load_dataset
from loguru import logger

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

llama2_prompt ={ "prompt_no_input":"""<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction} [/INST]"""}

TEXT_COLUMN = "text"

def process_data(data, tokenizer, job_config):
    data = data.to_pandas()
    data = data.fillna("")

    data = data[[TEXT_COLUMN]]
    if job_config.add_eos_token:
        data[TEXT_COLUMN] = data[TEXT_COLUMN] + tokenizer.eos_token
    data = Dataset.from_pandas(data)
    return data

def tokenize(tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id and add_eos_token:
        if len(result["input_ids"]) >= tokenizer.model_max_length:
            result["input_ids"] = result["input_ids"][:-1]
            result["attention_mask"] = result["attention_mask"][:-1]
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
    
def load_datasets(data_path):
    files = os.listdir(data_path)
    data_paths = [os.path.join(data_path,f) for f in files]
    all_datasets = []
    for file in data_paths:
        if not (file.endswith(".json") or file.endswith(".jsonl")):
            continue
        raw_dataset = load_dataset("json", data_files=file,cache_dir="./.cache/huggingface/datasets")
        
        all_datasets.append(raw_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets
    
def tokenize_func(examples,tokenizer,add_eos_token=True):
    text_all = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input = examples["input"][i]
        output = examples['output'][i]
        if input:
            instruction = instruction + input
            
        if output:
            context = llama2_prompt['prompt_no_input'].format_map({"instruction": instruction})+ " "+ output
        else:
            context = instruction
        if add_eos_token and not context.endswith(tokenizer.eos_token):
            context = context + tokenizer.eos_token
        if not context.startswith(tokenizer.bos_token):
            context = tokenizer.bos_token + context
        tokenized_output = tokenizer(context,add_special_tokens=False)
        
        text_all.append(tokenized_output)
    
    res = Dataset.from_list(text_all)
    return res.to_dict()


def group_texts(examples, block_size):
    # print(f"group_texts={examples}")
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result



class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
#         print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        print("Saving PEFT checkpoint at end")
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def train(
    base_model: str="meta-llama/Llama-2-7b-hf",
    data_path: str = "data/alapa",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 12,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 200,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    peft_path='',
    report_to='tensorboard',
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # wandb params
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    gradient_accumulation_steps=8,
    block_size: int = 512,
    use_int8 = False,
    add_eos_token = True,
    flash_attn=False
    ):

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LlaMA2-QLoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"peft_path: {peft_path}\n"
            f"block_size: {block_size}\n"
            f"add_eos_token:{add_eos_token}\n"
            f"use_int8:{use_int8}\n"
            f"flash_attn:{flash_attn}\n"
        )

    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)


    model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                load_in_8bit=use_int8,
                device_map=device_map,
                 use_cache=False,
            )

    
    if tokenizer.model_max_length > 16000:
        tokenizer.model_max_length = 16000

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if flash_attn:
        from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(len(tokenizer))

     # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if use_int8:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, peft_config)
    
    
    if peft_path: 
        adapters_weights = torch.load(f"{peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)

    model.print_trainable_parameters()

    print("start load datasets")
    
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
       
    # else:
    #     data = load_dataset(data_path,cache_dir="./.cache/huggingface/datasets")
        
    data=load_datasets(data_path)
    print(f"end load datasets,sample:{len(data['train'])}")


    def generate_promt_func(examples):
        
        return tokenize_func(examples,tokenizer,add_eos_token)



    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
        
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    
        
    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].map(generate_promt_func,
                                                      batched=True,
                                                      num_proc=4,)
        train_data = train_data.map(
            group_texts,
            batched=True,
            num_proc=4
        )
       
        val_data = train_val["test"].map(generate_promt_func,
                                                      batched=True,
                                                      num_proc=4,)
        val_data = val_data.map(
            group_texts,
            batched=True,
            num_proc=4
        )
        
    else:
        train_data = data.shuffle().map(generate_promt_func,
                                                      batched=True,
                                                      num_proc=8,)
        train_data = train_data.map(
            group_texts,
            batched=True,
            num_proc=8
        )
        val_data = None
    

    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        
    
    training_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        logging_steps=200,
        save_total_limit=5,
        warmup_steps=10,
        sharded_ddp=True,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=report_to,
        auto_find_batch_size=True,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=0.3,
        ddp_timeout=7200,
        fp16=True,
        group_by_length=group_by_length,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    )
    
    args = TrainingArguments(**training_args)

    data_collator = default_data_collator
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    model.config.use_cache = False
    trainer.add_callback(SavePeftModelCallback)
    
    train_result=trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model()
    print("finished save model")
    print("ignore the error at end")


if __name__ == "__main__":
    fire.Fire(train)
