import jsonlines
import json
def jwrite(data_path,data:list):
    with open(data_path,'w') as f:
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
    with open(data_path,'r') as f:
        data = json.load(f)
    print(f"total samples:{len(data)}")
    return data
