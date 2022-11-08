# 2022 LG Uplus AI Ground competition

- 관련 링크: https://stages.ai/competitions/208/overview/description
- 대회 설명: 2022.11.07 ~ 2022.12.02 동안 다음에 시청할 콘텐츠를 추천하는 테스크
- 대회 참여 목적
  - 실제 데이터를 통해서 추천모델을 만들어보고 평가까지 해보기
  - 딥러닝 모델을 만들면서 그 전에 해보지 못했던 과정 경험
    - hp tuning
    - mlflow
    - 성능과 속도 고도화 tip
    - torch 모델 시각화

# 진행 과정

### DeepFM baseline

- `history.drop_duplicates(['profile_id','log_time','album_id'])`으로 진행
- `meta`데이터에서 `sub_title`, `genre_(large, mid)` 컬럼 사용
- `profile`데이터 사용

### 그 이후 추가적으로 feature 추가해보기 (컨텐츠 시청시간, 컨텐츠 시청횟수)

### Boosting 모델 사용

### 모델 성능

|      Model       | 성능지표1 | 성능지표2 |
| :--------------: | :-------: | :-------: |
| DeepFM(baseline) |     0     |     0     |

# Reference

- paper
- MLflow
  - [MLflow를 이용한 머신러닝 프로젝트 관리. 박선준- PyCon Korea 2021](https://www.youtube.com/watch?v=H-4ZIfOJDaw)
