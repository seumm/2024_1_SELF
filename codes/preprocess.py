import json

def add_labels_from_jsonl(original_file_path, new_file_path, category):
    new_labels_list = []  # 레이블을 저장할 리스트
    with open(original_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            labels = data.get('labels', [])  # 'labels' 키에 해당하는 값을 가져옴
            # always, never을 포함하는 지 여부에 따라 분류
            if category[0] in labels or category[4] in labels:
                if category[0] == labels[0] and category[0] == labels[1] and category[0] == labels[2]:
                    data['label'] = 0
                    new_labels_list.append(data)
                elif category[4] == labels[0] and category[4] == labels[1] and category[4] == labels[2]:
                    data['label'] = 2
                    new_labels_list.append(data)
            else:
                data['label'] = 1
                new_labels_list.append(data)

    # 새로운 데이터를 JSONL 형식으로 파일에 씀
    with open(new_file_path, 'w', encoding='utf-8') as output_file:
        for new_data in new_labels_list:
            json.dump(new_data, output_file)
            output_file.write('\n')

file_path = './data/twentyquestions-all.jsonl'
new_file_path = './data/twentyquestions-all-preprocess-2.jsonl'
category = ["always", "usually", "sometimes", "rarely", "never"]

add_labels_from_jsonl(file_path, new_file_path, category)