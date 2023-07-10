# Likelihood(MLE와 MAP)
## 1. 확률 변수로서의 모델 파라미터
## 2. posterior와 prior, likelihood 사이의 관계
```text
[시나리오 예시]
- 데이터의 집합 X
- 데이터의 확률 분포 p(X)
- 목표: y = ax + b = θ^{⊤}x
```
위와 같은 조건일 때,</br>
- **prior** (prior probability) : 파라미터 공간에 주어진 확률 분포 p($\theta$); 사전 확률

- **likelihood (기능도, 우도)** : prior 분포가 고정되었을 때, 가지고 있는 데이터가 관찰될 확률

$$ p(X = x|\theta) $$

- MLE (maximum likelihood estimation) : 데이터들의 likelihood 값을 최대화하는 방향으로 모델을 학습시키는 방법