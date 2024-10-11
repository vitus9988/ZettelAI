from openai import OpenAI
import pandas as pd
import json
from collections import defaultdict
import itertools


key = "OPENAI APIKEY"
client = OpenAI(api_key=key)


dataset = pd.read_csv('./dataset/sample_unrefine_dataset.csv')
data = dataset.groupby(['main_topic', 'sub_topic']).head(1).reset_index(drop=True)

topic_dict = defaultdict(list)

for index, row in data.iterrows():
    key = (row['main_topic'], row['sub_topic'])
    topic_dict[key].append(row['content'])
    
memo_pairs = []

topic_keys = list(topic_dict.keys())

for key1, key2 in itertools.combinations(topic_keys, 2):
    main_topic1, sub_topic1 = key1
    main_topic2, sub_topic2 = key2
    
    if (main_topic1 != main_topic2) and (sub_topic1 != sub_topic2):
        memos1 = topic_dict[key1]
        memos2 = topic_dict[key2]

        for memo1 in memos1:
            for memo2 in memos2:
                memo_pairs.append((key1, memo1, key2, memo2))

jsonl_file_path = './dataset/batch_queue.jsonl'
count = 1
with open(jsonl_file_path, 'w', encoding='utf-8') as file:
    for (memo1, memo2) in memo_pairs:
        prompt = f"""아래의 두 메모의 메모내용을 읽고 파악하여 각 메모로부터 키워드를 찾고 이를 통해 새로운 개념이나 아이디어를 200글자 이하로 도출하세요.
    추출한 키워드에 대한 설명없이 새로운 개념이나 아이디어만을 출력하세요.\n
    [1번 메모]

    <메모내용>: {memo1}

    [2번 메모]

    <메모내용>: {memo2}
    """
        jsonl_format = {"custom_id": f"request-{count}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "You have a role in generating data to fine-tune other language models."},{"role": "user", "content": f"{prompt}"}],"max_tokens": 500}}
        file.write(json.dumps(jsonl_format,ensure_ascii=False) + '\n')
        count += 1
    
create_respones = client.files.create(
    file=open(jsonl_file_path, "rb"),
    purpose="batch"
    )

batch_res = client.batches.create(
    input_file_id=create_respones.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

client.batches.retrieve(batch_res.id)

