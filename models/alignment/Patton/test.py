import json

def count_missing_labels(file_path):
    missing_label_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            if 'label' not in data:
                missing_label_count += 1

    return missing_label_count

# 调用函数并打印结果
file_path = 'data/cora/nc-coarse/8_8/test.text.jsonl'
missing_count = count_missing_labels(file_path)
print(f"缺少 'label' 键的样本数量: {missing_count}")