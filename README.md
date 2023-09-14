# llama2-chat
#### data format
### 1.for instruction tuning data format is like 
```

data={"instruction":"","input":"","output":""}
prompt = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

{instruction} [/INST]"""
prompt
```

### 2.for chat bot data format is 
```
data={"instruction":"","input":"","output":""}
data['instruction']="""<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_2 }} </s><s>[INST] {{ user_msg_3 }} [/INST]"""
```
#### in my llm_trainer.py the input is not padding with -100. It's a bit like pretraining steps. from left to right to predict next word.
#### if you want to padd the input with -100.you can also use sft_train.sh
### 3.implement trl pkg SFTTrainer class
```
sh run_sfttrainer.sh
```
Huggingface link [My Huggingface Model](https://huggingface.co/Moses25/Llama2-Moses-7b-chat)
