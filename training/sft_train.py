import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List
import json
import transformers
from build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
def jload(data_path:str)-> List:
    with open(data_path,'r') as f:
        data = json.load(f)
    return data

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer,EarlyStoppingCallback,BitsAndBytesConfig


import bitsandbytes as bnb

import json
import os.path as osp
from typing import Union




ORCA_PROMPT_DICT={"prompt_no_input":(
    "### System:\n"
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    "### User:\n"
    "{instruction}"
    "\n\n### Response:"
),
"prompt_input":(
    "### System:\n"
    "You are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n"
    "### User:\n"
    "{instruction}"
    "\n\n### Input:\n"
    "{input}"
    "\n\n### Response:"
)}

llama2_prompt ={ "prompt_no_input":"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction} [/INST]"""}

header = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
)

ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
        }



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
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "data/alapa",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 12,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 200,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    cache_dir=None,
    peft_path='',
    report_to='none',
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    model_type="llama",
    load_int8=False,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
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
            f"cache_dir: {cache_dir}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"peft_path: {peft_path}\n"
            f"model_type: {model_type}\n"
            f"load_int8: {load_int8}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-2-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir=cache_dir,
       
    )
    if load_int8:
        print("model trainging for load_in8_bit")
        model = prepare_model_for_kbit_training(model)
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
            
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
        
    tokenizer.padding_side = "right"  # Allow batched inference
    
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
   
        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, config)
    
    
    if peft_path: 
        adapters_weights = torch.load(f"{peft_path}/adapter_model.bin")
        set_peft_model_state_dict(model, adapters_weights)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
   

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_data = build_instruction_dataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_seq_length=cutoff_len,
                data_cache_dir = None,
                preprocessing_num_workers = 4)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            sharded_ddp=True,
            logging_steps=200,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            ddp_timeout=3600,
            logging_strategy='steps',
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=report_to,
           

        ),
        data_collator=data_collator,
       
    )
    model.config.use_cache = False
    trainer.add_callback(SavePeftModelCallback)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train_result=trainer.train(resume_from_checkpoint=None)

#     model.save_pretrained(output_dir)
#     trainer.save_model(output_dir)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(output_dir)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)

