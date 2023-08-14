import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union, List
import datasets
import torch
import logging
from datasets import load_dataset, concatenate_datasets
import copy
import transformers
import random
from functools import partial

IGNORE_INDEX = -100

logger = logging.getLogger('__name__')


PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    )

def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                train_on_inputs=False,
                ):

    def tokenization(examples,train_on_inputs=False):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output in zip(examples['instruction'],examples['input'],examples['output']):
            if input is not None and input !="":
                instruction = instruction+'\n'+input
            source = prompt.format_map({'instruction':instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor((s + t)[:max_seq_length])
            labels = torch.LongTensor(([IGNORE_INDEX] * len(s) + t)[:max_seq_length])
            assert len(input_ids) == len(labels)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        results = {'input_ids':all_input_ids, 'labels': all_labels}
        if train_on_inputs:
            
            results = {'input_ids':all_input_ids, 'labels': input_ids}
        return results


    logging.info("building dataset...") 
    all_datasets = []
    files = os.listdir(data_path)
    data_paths = [os.path.join(data_path,f) for f in files]
    
    for file in data_paths:
        if not (file.endswith(".json") or file.endswith(".jsonl")):
            continue
        raw_dataset = load_dataset("json", data_files=file,cache_dir="./.cache/huggingface/datasets")
        tokenization_func = partial(tokenization,train_on_inputs=train_on_inputs)
        tokenized_dataset = raw_dataset.map(
                    tokenization_func,
                    batched=True,
                    num_proc=preprocessing_num_workers,
                    # remove_columns=["instruction","input","output"],
                    keep_in_memory=False,
                    desc="preprocessing on dataset",
                )

        processed_dataset = tokenized_dataset
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    
    all_datasets = concatenate_datasets(all_datasets)
    logger.info(f"**********{all_datasets[0]['input_ids']}")
    return all_datasets

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        
        
        input_ids_pt = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        
        labels_pt = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            
        return dict(
            input_ids=input_ids_pt,
            labels=labels_pt,
            attention_mask=input_ids_pt.ne(self.tokenizer.pad_token_id),
            )
