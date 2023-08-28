# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig,PeftModel,set_peft_model_state_dict,get_peft_model_state_dict,get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    
    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})

    num_epochs:Optional[int] = field(default=3, metadata={"help": "the train epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=200, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    block_size: Optional[int] = field(default=512, metadata={"help": "the block size parameter"})

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "the logging frequency"})
    peft_path: Optional[str] = field(default="", metadata={"help": "lora path"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    prompt_template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help as much as you can.
<</SYS>>

{instruction} [/INST]"""
    inst = example['instruction']
    input = example['input']
    output = example['output']
    if input:
        inst = inst + input
        
    text = prompt_template.format_map({"instruction":inst}) 
    if output:
        text = f"{text} {output}"
        
    return text


def alpaca_prompt_txt(example):
    pass

def group_texts(examples, block_size=512):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        total_length = 0
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def create_datasets(tokenizer, args):
    if args.dataset_name.endswith(".json") or args.dataset_name.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=args.dataset_name)
       
    else:
        dataset = load_dataset(args.dataset_name,cache_dir="/root/workspace/llama/.cache/huggingface/datasets")
    
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset['train'].train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        eos_token_id = tokenizer.eos_token_id,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        eos_token_id = tokenizer.eos_token_id,
        chars_per_token=chars_per_token,
    )
    

    # if args.block_size is None:
    #     block_size = tokenizer.model_max_length
    #     if block_size > 1024:
    #         logger.warning(
    #             "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
    #             " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
    #             " override this default with `--block_size xxx`."
    #         )
    #         block_size = 1024
    # else:
    #     if args.block_size > tokenizer.model_max_length:
    #         logger.warning(
    #             f"The block_size passed ({config.block_size}) is larger than the maximum length for the model"
    #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
    #         )
    #     block_size = min(args.block_size, tokenizer.model_max_length)
    

    # group_texts_fn = partial(group_texts, block_size)

    
    # train_dataset = train_dataset.map(
    #         group_texts_fn,
    #         batched=True,
    #         num_proc=4,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )

    # valid_dataset = valid_dataset.map(
    #         group_texts_fn,
    #         batched=True,
    #         num_proc=4,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )

    
    return train_dataset, valid_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    # gradient_accumulation_steps = gradient_accumulation_steps // world_size

base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        # quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        use_cache=False,
    )
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


base_model = get_peft_model(base_model, peft_config)

if script_args.peft_path:
    adapters_weights = torch.load(f"{script_args.peft_path}/adapter_model.bin")
    set_peft_model_state_dict(base_model, adapters_weights)

base_model.print_trainable_parameters()
        
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    num_train_epochs=script_args.num_epochs,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.report_to,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    save_total_limit=5,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_llama2",
)

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # peft_config=peft_config,
    packing=True,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)

for name, module in trainer.model.named_modules():
        # if isinstance(module, LoraLayer):
        #     if script_args.bf16:
        #         module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        # if "lm_head" in name or "embed_tokens" in name:
        #     if hasattr(module, "weight"):
        #         if script_args.bf16 and module.weight.dtype == torch.float32:
        #             module = module.to(torch.bfloat16)


trainer.train()
trainer.save_model(script_args.output_dir)
print("finished save model")
