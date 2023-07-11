# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 김민식
- 리뷰어 : 황인준


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
 
- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  > - 만일 모든 데이터 컬럼(특징)을 넣는다면 오차 값이 말도 안 되게 적게 나올 수도 있습니다
- [O] 코드가 에러를 유발할 가능성이 없나요?
  >
- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 
- [O] 코드가 간결한가요?
  ```python
  fig, ax = plt.subplots(2, 3, constrained_layout=True)
  sns.countplot(x=train['datetime'].dt.year, ax=ax[0][0])
  sns.countplot(train['datetime'].dt.month, ax=ax[0][1])
  sns.countplot(train['datetime'].dt.day, ax=ax[0][2])
  sns.countplot(train['datetime'].dt.hour, ax=ax[1][0])
  sns.countplot(train['datetime'].dt.minute, ax=ax[1][1])
  sns.countplot(train['datetime'].dt.second, ax=ax[1][2])
  ```

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python

```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
