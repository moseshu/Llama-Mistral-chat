# llama_v2_prompt(a)
import jsonlines
import json
def jwrite(data_path,data:list):
    with open(data_path,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

    
def write_jsonstr(data_path,data_list:list):
    with jsonlines.open(data_path,"a") as f:
        for i in data_list:
            # data = json.dumps(i,ensure_ascii=False,indent=2)
            f.write(i)



def load_jsonstr1(data_path,nums=2e10,line_num=0):
    lines = []
    count = 1
    f = open(data_path)
    line = f.readline().strip()
    # print(line)
    while count <= line_num:
        line = f.readline().strip()
        count += 1
    num_count = 0
    while line:
        lines.append(line)
        line = f.readline()
        if num_count >= nums:
            break
        num_count += 1
    f.close()
    # with open(data_path,'r') as f:
    #     lines = f.read().splitlines()
    data = []
    error = 0
    for item in lines:
        try:
            data.append(json.loads(item))
        # data = [json.loads(i) for i in lines]
        except:
            error += 1
            continue
    print(f"error:{error},total samples:{len(data)}")
    return data


from typing import List
import json
def jload(data_path:str)-> List:
    if data_path.endswith("json"):
        with open(data_path,'r') as f:
            data = json.load(f)
        print(f"total samples:{len(data)}")
        return data
    else:
        return load_jsonstr1(data_path)



#将mistral或者llama2的多轮对话的数据格式转换成llama3
def change_template(data:list):
    """ 
    instruction 有多轮对话的格式 含有 [INST] [/INST] <<SYS>> <</SYS>> <s> </s>
    :param data: data=[{"instruction":"","input":"","output":""}]
    """
    for content in data:
        # if "[INST]" in content['instruction'] and "[/INST]" in content['instruction'] and "<<SYS>>" in content['instruction']:
        if not content['output']:
            text = content['instruction'].lstrip("<s>").rstrip("</s>")
            filter_text = content['instruction'].lstrip("<s>").rstrip("</s>").replace("[/INST] ","[/INST]").replace("[INST] ","[INST]").replace("</s><s>[INST]","<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n").replace("[/INST]","<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n").replace('<<SYS>>\n',"<|start_header_id|>system<|end_header_id|>\n\n").replace("\n<</SYS>>\n\n","<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n").lstrip("[INST]").strip()
            if "<<SYS>>" in text and "[INST]" in text:
                text = "<|begin_of_text|>" + filter_text
                
            else:
                if "[INST]" in text:
                    text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + filter_text
            
            content['instruction'] = text

            
    return data



def chat_format(conversation:list,tokenizer,chat_type="mistral",system_prompt=None):
    """ 
    :param conversation:
    >>conversation =[{"role":"user","content":"你好"},
       {"role":"assistant","content":"dsf"},
         {"role":"user","content":"你sdfs好"},]
    :param tokenizer: 
    >> tokenizer = AutoTokenizer.from_pretrained(f"Meta-Llama-3-8B-Instruct")
    >> content_prompt = chat_format(conversation,tokenizer,"llama3")
    """
    mistral_template="{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
    llama3_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    qwen_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    if not system_prompt:
        system_prompt = "You are a helpful, respectful and honest assistant.Help humman as much as you can."
    ap = [{"role":"system","content":system_prompt}] + conversation
    if chat_type=='mistral':
        id = tokenizer.apply_chat_template(ap,chat_template=mistral_template,tokenize=False)
    elif chat_type=='llama3':
        id = tokenizer.apply_chat_template(ap,chat_template=llama3_template,tokenize=False)
        id = id.rstrip("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n") + "<|eot_id|>"
    elif chat_type=='qwen':
        id  = tokenizer.apply_chat_template(ap,chat_template=qwen_template,tokenize=False)+"<|im_end|>"
    # text = tokenizer.decode(id)
    return id
