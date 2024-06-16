import os
import sys
from typing import List
import fire
import torch
import datasets
import transformers
from datasets import load_dataset,Dataset,concatenate_datasets
import json
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from itertools import chain
from datasets import Dataset, load_dataset
from loguru import logger
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch
from llama_xformers_attn_monkey_patch import replace_llama_attn_with_xformers_attn
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
BitsAndBytesConfig,
)
import bitsandbytes as bnb

###
# apply_attention_patch(use_memory_efficient_attention=True)
# apply_ntk_scaling_patch(2.0)
replace_llama_attn_with_xformers_attn()
###

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"



llama2_prompt ={ "prompt_no_input":"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help humman as much as you can.
<</SYS>>

{instruction} [/INST]"""}

llama3_prompt="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"


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

class Concatenator(object):
    def __init__(self, chunk_size=4096):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


def load_datasets(data_path):
    files = os.listdir(data_path)
    data_paths = [os.path.join(data_path,f) for f in files]
    all_datasets = []
    for file in data_paths:
        if not (file.endswith(".json") or file.endswith(".jsonl")):
            continue
        try:
            raw_dataset = load_dataset("json", data_files=file,cache_dir="./.cache")
            print(f"loading file:{file}")
            cmd = f'rm -r .cache/'
            all_datasets.append(raw_dataset['train'])
            # subprocess.check_call(cmd, shell=True)
        except:
            print(f'load {file}file error check your file is correct json file')
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

def tokenize_func(examples,tokenizer,add_eos_token=True):
    text_all = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input = examples["input"][i]
        output = examples['output'][i]
        if input:
            instruction = instruction + "\n" + input
            
        if output:
            context = llama2_prompt['prompt_no_input'].format_map({"instruction": instruction})+ " "+ output
        else:
            context = instruction
        if add_eos_token and not context.endswith(tokenizer.eos_token):
            context = context + tokenizer.eos_token
        if not context.startswith("<s>"):
            context = tokenizer.bos_token + context 
        tokenized_output = tokenizer(context,add_special_tokens=False)
        
        # if len(tokenized_output['input_ids']) > 0:
        text_all.append(tokenized_output)
    
    res = Dataset.from_list(text_all)
    return res.to_dict()



def llama3_tokenize_func(examples,tokenizer,add_eos_token=True):
    text_all = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input = examples["input"][i]
        output = examples['output'][i]
        if input:
            instruction = instruction + "\n" + input
            
        if output:
            context = llama3_prompt.format_map({"instruction": instruction})+ "<|start_header_id|>assistant<|end_header_id|>"+ output
        else:
            context = instruction
        if add_eos_token and not context.endswith("<|eot_id|>"):
            context = context + "<|eot_id|>"
        if not context.startswith("<|begin_of_text|>"):
            context = "<|begin_of_text|>" + context
        tokenized_output = tokenizer(context,add_special_tokens=False)
        # if len(tokenized_output['input_ids']) > 0:
        text_all.append(tokenized_output)
    
    res = Dataset.from_list(text_all)
    return res.to_dict()

def find_all_linear_names(model):
  
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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
    gradient_accumulation_steps=16,
    block_size: int = 4096,
    use_int8 = False,
    use_int4=False,
    add_eos_token = True,
    flash_attn=False,
    chat_type="mistral",
    use_rslora=False,
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
            f"use_int4:{use_int4}\n"
            f"flash_attn:{flash_attn}\n"
            f"chat_type:{chat_type}\n"
            f"use_rslora:{use_rslora}\n"
        
        )

    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if use_int4:
        use_int8 = False
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_int4,
            load_in_8bit=use_int8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
    )
    
    if use_int8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=use_int8)
        
    model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                # load_in_8bit=use_int8,
                # load_in_4bit=use_int4,
                device_map=device_map,
                use_cache=False,
                quantization_config=bnb_config if (use_int8 or use_int4) else None,
                # use_flash_attention_2=True,
                attn_implementation="flash_attention_2",
            )

    
        
    if tokenizer.model_max_length > 1280000:
        tokenizer.model_max_length = 1280000

    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({"pad_token":DEFAULT_PAD_TOKEN})
        tokenizer.pad_token_id = 0
        # tokenizer.pad_token="[PAD]"
        


    if flash_attn:
        from llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()
    
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(len(tokenizer))

    if use_int8 or use_int4:
        model = prepare_model_for_kbit_training(model)
    #
    if "mixtral" in base_model.lower() :
        model.config.output_router_logits = True
        if use_int4:
            lora_target_modules = find_all_linear_names(model) 
        print(f"use_int4 lora_target_modules:{lora_target_modules}")
    
     # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )
    # if not peft_path:
    model = get_peft_model(model, peft_config)

    
    
    if peft_path:
        adapters_weights = torch.load(f"{peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)
        # model = PeftModel.from_pretrained(model, peft_path,is_trainable=True,torch_dtype=torch.float16,device_map="auto")
        
        
    
    
    model.print_trainable_parameters()

    
    print("start load datasets")
    
    # if data_path.endswith(".json") or data_path.endswith(".jsonl"):
    #     data = load_dataset("json", data_files=data_path)
       
    # else:
    #     data = load_dataset(data_path,cache_dir="./.cache/huggingface/datasets")
    
    # print(f"end load datasets,sample:{len(data['train'])}")
    data=load_datasets(data_path)
    gpu_nums = torch.cuda.device_count()
    print(f"end load datasets,sample:{len(data)}\ngpu_nums:{gpu_nums}")
    
    max_steps = len(data) / (gpu_nums * gradient_accumulation_steps * micro_batch_size)
    print(f"max_steps is:{max_steps}")
    
    def generate_promt_func(examples):
        
        if chat_type=='mistral':
            return tokenize_func(examples,tokenizer,add_eos_token)
        elif chat_type=='llama3':
            return llama3_tokenize_func(examples,tokenizer,add_eos_token)
        else:
            print("error")
            return ""



    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 8192:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 8192
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
        
    
    
        
    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_promt_func,
                                                      batched=True,
                                                      num_proc=4,)
        train_data = train_data.map(
           Concatenator(chunk_size=block_size),
            batched=True,
            num_proc=4
        )
       
        val_data = train_val["test"].shuffle().map(generate_promt_func,
                                                      batched=True,
                                                      num_proc=8,)
        val_data = val_data.map(
            Concatenator(chunk_size=block_size),
            batched=True,
            num_proc=4
        )
        
    else:
        # torch.manual_seed(1)
        train_data = data.map(generate_promt_func,
                            batched=True,
                            num_proc=8,
                            remove_columns=list(data.features)
                             )
        print(f"tokenized train data samples:{len(train_data)}")
        # random_seed = 1
        # torch.manual_seed(random_seed)
        train_data = train_data.shuffle(seed=1).map(
            Concatenator(chunk_size=block_size),
            batched=True,
            num_proc=8
        )
        val_data = None
        print(f"train data samples:{len(train_data)}")

    # print(f"\ntotal tokens is: {total_tokens / 1000000000}B")
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
        save_steps=200,
        save_total_limit=3,
        warmup_steps=10,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=report_to,
        auto_find_batch_size=False,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_steps = -1,
        max_grad_norm=0.3,
        ddp_timeout=7200,
        fp16=False,
        bf16=True,
        group_by_length=group_by_length,
        fsdp="full_shard",
        fsdp_config={
            "backward_prefetch": "backward_pre",
  "forward_prefetch": "false",
  "use_orig_params": "false",
        },
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False #if ddp else None,
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

    # for name, module in trainer.model.named_modules():
    #     # if isinstance(module, LoraLayer):
    #     #     if script_args.bf16:
    #     #         module = module.to(torch.bfloat16)
    #     if "norm" in name:
    #         module = module.to(torch.float32)
        # if "lm_head" in name or "embed_tokens" in name:
        #     if hasattr(module, "weight"):
        #         if script_args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)

    
    train_result=trainer.train(resume_from_checkpoint=None)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model()
    print("finished save model")


if __name__ == "__main__":
    fire.Fire(train)
