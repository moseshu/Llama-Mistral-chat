# Mistral and Llama3 chat
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



mistral_template="{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

llama3_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"

def chat_format(conversation:list,tokenizer,chat_type="mistral"):
    system_prompt = "You are a helpful, respectful and honest assistant.Help humman as much as you can."
    ap = [{"role":"system","content":system_prompt}] + conversation
    if chat_type=='mistral':
        id = tokenizer.apply_chat_template(ap,chat_template=mistral_template,tokenize=False)
    elif chat_type=='llama3':
        id = tokenizer.apply_chat_template(ap,chat_template=llama3_template,tokenize=False)
        #id = id.rstrip("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    return id

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
Huggingface link [My Huggingface Chat Llama3 Moldel 8B](https://huggingface.co/Moses25/Llama-3-8B-chat-32K))

![image](https://github.com/moseshu/llama2-chat/assets/23112888/01d729fe-5c1c-44f1-a55e-33c20a8b5a79)

### 5.FSDP
refer to [llama-recipes](https://github.com/meta-llama/llama-recipes)
