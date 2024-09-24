import pandas as pd
from tqdm import tqdm
from openai import OpenAI


key = "OPENAI API KEY"
client = OpenAI(api_key=key)


data = pd.read_csv("./dataset/sample_unmeasured_dataset.csv")
data['value'] = 0.0

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="아이디어 점수 측정 중"):
    prom = f"""다음은 서로 다른 <메모내용>을 바탕으로 새로운 <아이디어>를 추론한 예시이다.
            추론의 타당성, 정확성, 창의성, 독창성을 각 10점 만점으로 평가하여 종합평균점수를 출력하라.
            단, 각 항목에 대한 점수측정 결과 없이 종합 평균 점수를 숫자만을 사용하여 실수형으로 출력하라
            (예시: 4.2)

            <메모내용>: {row['instruction']}
            <아이디어>: {row['output']}
            """
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
    try:
        score = float(answer)
    except ValueError:
        score = None
    data.at[index, 'value'] = score
    
data.to_csv('./dataset/sample_measured_dataset.csv', index=False)