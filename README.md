# ZettelAI
[2024 Gemma 파인튜닝톤 (아이디어톤)](https://aifactory.space/task/2733/overview)에 참가한 "사용자 메모 데이터를 활용한 제텔카스텐 구현" 관련 소스코드 저장소입니다.

## 데이터 생성

1. 사전에 구성한 대주제 및 소주제 entity를 사용, gpt-4o-mini API를 통해 [초기 데이터셋](https://huggingface.co/datasets/vitus9988/ko_gpt4omini_note_15.4k)을 구축합니다. (make_dataset_by_entity.py)
2. 1에서 생성한 데이터셋에서 대주제와 소주제가 다른 데이터쌍을 만들고 [메모A, 메모B - 아이디어]와 같은 instruction - output 형태의 데이터셋으로 변환합니다. (make_instruction_dataset.py)
3. 2에서 변환한 데이터셋에서 다시 gpt-4o-mini API를 사용하여 instruction - output 관계의 적절성을 점수로 변환하여 value 컬럼에 추가합니다. (calc_instruction_dataset.py)
   
## 모델 학습

google/gemma-2-9b-it으로 qlora 학습을 진행합니다.
학습에 사용하는 데이터셋은 '데이터 생성' 3에서 진행한 value 값을 사분위수(IQR)를 통해 버림처리 할 값을 얻고 해당 수치 이상의 데이터만을 sklearn 라이브러리를 사용하여 8:2로 train, test 데이터로 나누어 학습 및 검증에 사용합니다.

