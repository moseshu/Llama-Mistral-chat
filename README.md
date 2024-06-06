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
Huggingface link [My Huggingface Chat Model 7B](https://huggingface.co/Moses25/Mistral-7B-chat-32k)

### 4.autotrain llama   [autotrain-advanced](https://github.com/huggingface/autotrain-advanced)
```
cd autotrain
sh run.sh
or
sh script/llama_sft.sh
```
Huggingface link [My Huggingface Chat Moldel 13B]([https://huggingface.co/Moses25/MosesLM-13B-chat](https://huggingface.co/Moses25/Meta-LlaMA-3-8B-Instruct-16k))

![image](https://github.com/moseshu/llama2-chat/assets/23112888/ab1b4a31-51d7-40b3-a6f5-306013a43acb)

