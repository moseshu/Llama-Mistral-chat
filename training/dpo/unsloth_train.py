from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer,DPOTrainer
from transformers import TrainingArguments
from datasets import  Dataset, load_dataset,concatenate_datasets
from dataclasses import dataclass, field
from typing import Optional,List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
import os
from unsloth import PatchDPOTrainer


llama2_prompt ={ "prompt_no_input":"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help humman as much as you can.
<</SYS>>

{instruction} [/INST]"""}

TEXT_COLUMN = "text"

def _pro_data(item:dict):
    instruction = item['instruction']
    input = item['input']
    output = item['output']
    if input:
        instruction = instruction + "\n" + input
    
    if output:
        res = llama2_prompt['prompt_no_input'].format_map({"instruction": instruction}) + " " + output
    else:
        res = instruction
        
    if res.startswith("<s>"):
        res = res.lstrip("<s>")
        
    if res.endswith("</s>"):
        res = res.strip("</s>")
        
    return res
    

def process_data(data, tokenizer,add_eos_token=True):
    data = data.to_pandas()
    data[TEXT_COLUMN] = data.apply(lambda x:_pro_data(dict(x)),axis=1)
    data = data.fillna("")

    data = data[[TEXT_COLUMN]]
    if add_eos_token:
        data[TEXT_COLUMN] = data[TEXT_COLUMN] + tokenizer.eos_token
    data = Dataset.from_pandas(data)
    return data


def load_datasets(data_path):
    files = os.listdir(data_path)
    data_paths = [os.path.join(data_path,f) for f in files]
    all_datasets = []
    for file in data_paths:
        if not (file.endswith(".json") or file.endswith(".jsonl")):
            continue
        try:
            raw_dataset = load_dataset("json", data_files=file,cache_dir=".cache")
            print(f"loading file:{file}")
            cmd = f'rm -r .cache/'
            
            all_datasets.append(raw_dataset['train'])
            # subprocess.check_call(cmd, shell=True)
        except:
            print(f'load {file}file error check your file is correct json file')
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


    

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
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=20, metadata={"help": "the saving frequency"})
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
    target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj")
    
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    report_to: Optional[str] = field(default="tensorboard", metadata={"help": "the logging frequency"})
    peft_path: Optional[str] = field(default="", metadata={"help": "lora path"})
    
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "4bit train"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


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

def train():
    
    max_seq_length = script_args.seq_length # Supports RoPE Scaling interally, so choose any!
    # Get LAION dataset
   
    dataset = load_datasets(script_args.dataset_name)
    
    
    # 4bit pre quantized models we support - 4x faster downloading!
    
    # Load Llama model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = script_args.model_name, # Supports Llama, Mistral - replace this!
        max_seq_length = script_args.seq_length,
        dtype = None,
        load_in_4bit = script_args.load_in_4bit,
    )
    model.use_cache=False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 2
    dataset = process_data(dataset,tokenizer)
    if model.get_input_embeddings().weight.size(0) != len(tokenizer):
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = script_args.lora_r,
        target_modules = script_args.target_modules.split(","),
        lora_alpha = script_args.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = script_args.gradient_checkpointing,
        random_state = 3407,
        max_seq_length = max_seq_length,)
    
    print_trainable_parameters(model)
    print(f"tokenizer pad_token:{tokenizer.pad_token}")
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = TEXT_COLUMN,
        max_seq_length = max_seq_length,
        dataset_num_proc=8,
        tokenizer = tokenizer,
        args = TrainingArguments(
            per_device_train_batch_size = script_args.per_device_train_batch_size,
            gradient_accumulation_steps = script_args.gradient_accumulation_steps,
            warmup_steps = script_args.num_warmup_steps,
            max_steps = script_args.max_steps,
            num_train_epochs = script_args.num_epochs,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = script_args.logging_steps,
            save_steps=script_args.save_steps,
            output_dir = script_args.output_dir,
            save_total_limit=3,
            learning_rate=script_args.learning_rate,
            report_to=script_args.report_to,
            optim = "adamw_8bit", #paged_adamw_32bit->fp16  adamw_8bit-> 4bit
            seed = 3407,
        ),
    )
    trainer.train()
    trainer.save_model(script_args.output_dir)
    
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

def train_dpo():
    # PatchDPOTrainer()
    model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = script_args.model_name, # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
                    max_seq_length = script_args.seq_length,
                    dtype = None,
                    load_in_4bit = script_args.load_in_4bit,
                    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model.use_cache=False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 2

    print(f"tokenizer pad_token:{tokenizer.pad_token}")
    
    model = FastLanguageModel.get_peft_model(
                    model,
                    r = script_args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                    target_modules =script_args.target_modules.split(",") ,
                    lora_alpha = script_args.lora_alpha,
                    lora_dropout = 0, # Currently only supports dropout = 0
                    bias = "none",    # Currently only supports bias = "none"
                    use_gradient_checkpointing = script_args.gradient_checkpointing,
                    random_state = 3407,
                    use_rslora = False,  # We support rank stabilized LoRA
                    # loftq_config = None, # And LoftQ
    )

    dataset = load_datasets(script_args.dataset_name)
    
    
    dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = script_args.per_device_train_batch_size,
        gradient_accumulation_steps = script_args.gradient_accumulation_steps,
        warmup_ratio = 0.1,
        num_train_epochs = script_args.num_epochs,
        learning_rate = script_args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = script_args.logging_steps,
        optim = "adamw_torch",#adamw_torch adamw_8bit
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 42,
        save_steps=script_args.save_steps,
        save_total_limit=3,
        report_to=script_args.report_to,
        output_dir =script_args.output_dir,
    ),
    beta = 0.05,
    train_dataset = dataset,
    # eval_dataset = raw_datasets["test"],
    tokenizer = tokenizer,
    max_length = script_args.seq_length,
    max_prompt_length = 512,
    )

    
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)
    
    # save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    
if __name__ == "__main__":
    # train()
    train_dpo()
