import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import json

key = "OPENAI API KEY"
client = OpenAI(api_key=key)

dataset = {"main_topic":[], "sub_topic":[], "content":[]}
df = pd.DataFrame(dataset)

with open("./dataset/sample_entity.json", 'r', encoding='utf-8') as file:
    data = json.load(file)
    
for main_topic in data:
    for sub_topic in tqdm(data[main_topic], desc=f"{main_topic} 메모 작성 중"):
        for _ in range(40):
            prom = f"""Given the following "{main_topic}" vs. topic, think of a "{sub_topic}" on your own Select either a positive or negative or Neutral viewpoint to view. and print out a 300-word or less text about it in Korean.
    No titles or descriptions of the text, just the text"""

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You have a role in generating data to fine-tune other language models."},
                    {
                        "role": "user",
                        "content": prom
                    } 
                ]
            )
            answer = completion.choices[0].message.content
            dataset['main_topic'].append(main_topic)
            dataset['sub_topic'].append(sub_topic)
            dataset['content'].append(answer)
            df = pd.DataFrame(dataset)
df.to_csv('./dataset/sample_unrefine_dataset.csv', index=False)


