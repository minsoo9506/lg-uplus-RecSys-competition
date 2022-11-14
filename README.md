# 2022 LG Uplus AI Ground competition

- 관련 링크: https://stages.ai/competitions/208/overview/description
- 대회 설명: 2022.11.07 ~ 2022.12.02 동안 다음에 시청할 콘텐츠를 추천하는 테스크
- 대회 참여 목적
  - 실제 데이터를 통해서 추천모델을 만들어보고 평가까지 해보기
  - 딥러닝 모델을 만들면서 그 전에 해보지 못했던 과정 경험
    - hp tuning (optuna): 근데 시간이 오래 걸릴 것 같아서 일단 후순위
    - mlflow
    - 성능과 속도 고도화 tip
    - lightning callback 함수 사용

# 진행 과정

### DeepFM baseline

- 사용 feature
  - `history.drop_duplicates(['profile_id','log_time','album_id'])`으로 진행
  - `meta`데이터에서 `sub_title`, `genre_(large, mid)` 컬럼 사용
  - `profile`데이터에서 `sex`, `age`, `pr_interest_keyword_cd_1`, `ch_interest_keyword_cd_1` 사용
- negative sampling
  - train에서만 negetive sampling 진행
  - 각 유저별 데이터별로 neg_ratio만큼 sampling 진행
  - sampling은 random하게 진행
- model
  - DeepFM 사용 (binary classification)

### `mlflow`

- 설치: `pip install mlflow`
- pytorch lightning과 함께 사용하기 때문에 좀 더 편하기 tracking할 수 있다.
- 아래와 같이 코드를 추가한다. (lightning에서 mlflow logger가 존재하므로 이를 이용해도 된다)

```python
    # mlflow
    mlflow.pytorch.autolog()

    # Train the model
    with mlflow.start_run() as run:
        trainer.fit(DeepFM_lit_model, train_loader, valid_loader)
```

### NDCG@K

- 1에 가까울수록 좋다.
- 순서별로 가중치값(relevance)를 다르게 적용한다

먼저, NDCG@K 값을 구성하는 요소들을 하나씩 알아보자.

- Relevance
  - 정해진 값은 아니고 user와 item간의 관련성을 수치로 표현한 것이다.
- Discounted Cumulative Gain (DCG)
  - 추천 item의 순서에 따라 가중치를 곱해서 relevance의 합을 구한 것이다.

$$DCG_K = \sum_{k=1}^K \frac{rel_i}{log(i+1)}$$

최종적으로 Normalize를 한 값이 NDGC@K인 것이다. 이때 IDGC는 가장 이상적인 경우의 DCG값을 구한 것이다.

$$NDCG_K = \frac{DCG}{IDCG}$$

### 모델 성능

|      Model       | 성능지표1 | 성능지표2 |
| :--------------: | :-------: | :-------: |
| DeepFM(baseline) |     0     |     0     |

# Reference

- [MLflow를 이용한 머신러닝 프로젝트 관리. 박선준- PyCon Korea 2021](https://www.youtube.com/watch?v=H-4ZIfOJDaw)
- [[추천시스템] 성능 평가 방법 - Precision, Recall, NDCG, Hit Rate, MAE, RMSE](https://sungkee-book.tistory.com/11)
