from datasets import load_dataset,concatenate_datasets,Dataset
import os
from tqdm import tqdm
from itertools import chain
import torch
from torch.utils.data import Dataset as TDataset
from dataclasses import dataclass


import transformers


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
    # print(f"data_file:{data_paths}")
    all_datasets = []
    for file in data_paths:
        if file.endswith(".json") or file.endswith(".jsonl"):
           
            try:
                raw_dataset = load_dataset("json", data_files=file,cache_dir=".cache")
                print(f"load dataset file:{file}")
                all_datasets.append(raw_dataset['train'])
                # subprocess.check_call(cmd, shell=True)
            except:
                print(f'load {file}file error check your file is correct json file')
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets

def get_custom_dataset1(dataset_config,tokenizer, split):
    dataset = load_datasets(dataset_config.data_path)
    max_seq_length=16384
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=dataset_config.max_seq_length,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result['labels'] = result["input_ids"].copy()
        
        
        return result
    def generate_and_tokenize_prompt(data_point):
        llama2_prompt ="""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help as much as you can.
<</SYS>>

{instruction} [/INST]"""
        input_ids = []
        labels = []
        for i  in range(len(data_point)):
            
            instruction = data_point['instruction'][i]
            input = data_point["input"][i]
            output = data_point['output'][i]
            if input:
                instruction = instruction + "\n" + input
                
            if output:
                context = llama2_prompt.format_map({"instruction": instruction})+ " "+ output
            else:
                context = instruction
            if not context.endswith(tokenizer.eos_token):
                context = context + tokenizer.eos_token
            if not context.startswith(tokenizer.bos_token):
                context = tokenizer.bos_token + context
    
            tokenized_full_prompt = tokenize(context)
            
            input_ids.append(torch.LongTensor(tokenized_full_prompt['input_ids']))
            labels.append(torch.LongTensor(tokenized_full_prompt['labels']))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
        res = Dataset.from_dict(dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        ))
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
    dataset = dataset.shuffle().map(generate_and_tokenize_prompt,num_proc=8,batched=True)
    data = dataset.train_test_split(test_size=dataset_config.test_size,seed=42,shuffle=True)
    return data['train'],data['test']

    

###********start********###
def get_custom_dataset(dataset_config, tokenizer, split):
 
    dataset = load_datasets(dataset_config.data_path)
        

    llama2_prompt ="""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help humman as much as you can.
<</SYS>>

{instruction} [/INST]"""

    def apply_prompt_template(sample):
        instruction = sample['instruction']
        input = sample['input']
        output = sample['output']
        if input:
            instruction = instruction + "\n" + input
        
        if output:
            res = llama2_prompt.format_map({"instruction": instruction})+ " "+ output
        else:
            res = instruction
        if not res.endswith(tokenizer.eos_token):
            res = res + tokenizer.eos_token
            
        if tokenizer.bos_token is not None and res.startswith(tokenizer.bos_token):
            res = res.lstrip(tokenizer.bos_token)
        # print(f"examples:{res}")
        
        return {
            "text": res
        }
    
    dataset = dataset.map(apply_prompt_template,num_proc=8, remove_columns=list(dataset.features))
    
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        num_proc=8,
        remove_columns=list(dataset.features),
    ).shuffle(seed=1).map(Concatenator(chunk_size=dataset_config.max_seq_length), num_proc=8,batched=True)
    # data = dataset.train_test_split(test_size=dataset_config.test_size,seed=42)
    
    return dataset #data['train'],data['test']

###********end********###

def get_custom_dataset2(dataset_config, tokenizer, split):
   
    dataset = load_datasets(dataset_config.data_path)
    

    llama2_prompt ="""[INST] <<SYS>>
You are a helpful, respectful and honest assistant.Help humman as much as you can.
<</SYS>>

{instruction} [/INST]"""

    def apply_prompt_template(sample):
        # print(f"sample:{sample.keys()}")
        text_all = []
        for i in range(len(sample['instruction'])):
            instruction = sample['instruction'][i]
            input = sample["input"][i]
            output = sample['output'][i]
            if input:
                instruction = instruction + "\n" + input
                
            if output:
                context = llama2_prompt.format_map({"instruction": instruction})+ " "+ output
            else:
                context = instruction
            if not context.endswith(tokenizer.eos_token):
                context = context + tokenizer.eos_token
            if not context.startswith(tokenizer.bos_token):
                context = tokenizer.bos_token + context 
            # print(f"context==:{context}")
            tokenized_output = tokenizer(context,add_special_tokens=False)
            tokenized_output["labels"] = tokenized_output["input_ids"].copy()
            # if len(tokenized_output['input_ids']) > 0:
            text_all.append(tokenized_output)
        res = Dataset.from_list(text_all)
        return res.to_dict()

    
    dataset = dataset.shuffle(seed=42).map(apply_prompt_template,num_proc=8,batched=True)
    
    
    data = dataset.train_test_split(test_size=dataset_config.test_size,seed=42,shuffle=True)
    return data['train'],data['test']




