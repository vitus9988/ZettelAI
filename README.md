# ZettelAI
LLM - 제텔카스텐 구현을 위한 데이터 생성 및 LLM 파인튜닝에 대한 기술


## 데이터 생성

1. 사전에 구성한 대주제 및 소주제 entity를 사용, gpt-4o-mini API를 통해 초기 데이터셋을 구축합니다. (make_dataset_by_entity.py)
2. 1에서 생성한 데이터셋에서 대주제와 소주제가 다른 데이터쌍을 만들고 [메모A, 메모B - 아이디어]와 같은 instruction - output 형태의 데이터셋으로 변환합니다. (make_instruction_dataset.py)
3. 2에서 변환한 데이터셋에서 다시 gpt-4o-mini API를 사용하여 instruction - output 관계의 적절성을 점수로 변환하여 value 컬럼에 추가합니다. (calc_instruction_dataset.py)


