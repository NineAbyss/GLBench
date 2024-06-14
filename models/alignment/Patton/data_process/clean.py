import json
from tqdm import tqdm
def clean_data(file_path):
    cleaned_data = []
    removed_count = 0  # 初始化被删除的数据计数器
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            if data['q_text'] :  # 检查q_text和k_text是否为空
                cleaned_data.append(data)
            else:
                removed_count += 1  # 如果数据不符合条件，计数器加一

    print(f"被清理的数据数量: {removed_count}")  
 
    with open('/home/yuhanli/GLBench/models/alignment/Patton/data/instagram/nc-coarse/8_8/test.jsonl', 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item) + '\n')
    return cleaned_data

# 使用函数
cleaned_data = clean_data('/home/yuhanli/GLBench/models/alignment/Patton/data/instagram/nc-coarse/8_8/test.jsonl')