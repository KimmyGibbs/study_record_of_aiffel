# 텍스트 데이터 다루기

## 1-1. 들어가며
**학습 내용**</br>
- 2. 전처리: 자연어의 노이즈 제거
    - 자연어에 포함된 대표적인 세 가지 노이즈 유형을 확인
    - 노이즈 해결 방법을 학습
- 3. 분산표현: 바나나와 사과의 관계를 어떻게 표현할까?
    - 분산표현과 희소표현이 무엇인지를 학습
- 4. 토큰화: 그녀는? 그녀+는?
    - 대표적인 토큰화 기법을 학습
        - 공백 기반 토큰화
        - 형태소 기반 토큰화
- 5. 토큰화: 다른 방법들
    - OOV 문제를 해결한 BPE, BPE를 변형한 WPM에 대해 학습
- 6. 토큰에게 의미를 부여하기
    - 토큰화 기법이 아닌 단어 간 의미를 찾는 대표적인 세 가지 유형을 학습

```shell
# 아래와 같이 작업디렉토리를 구성하였음
$ mkdir -p ~/aiffel/text_preprocess
```

[자연 언어와 프로그래밍 언어](https://dukeyang.tistory.com/2)</br>
[Google's Natural Language Parser Model-SyntaxNet](https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html)</br>

---

## 1-2. 전처리 : 자연어의 노이즈 제거
**노이즈 유형 (1) 문장부호 : Hi, my name is john.**</br>
&nbsp;</br>
`Hi,`의 경우를 해결하기 위해서(문장부호와 단어가 같이 쓰인 상황) **문장부호 양쪽에 공북을 추가**해서 노이즈를 제거한다.</br>

```python
def pad_punctuation(sentence, punc):
    for p in punc:
        sentence = sentence.replace(p, " " + p + " ")

    return sentence

sentence = "Hi, my name is john."

print(pad_punctuation(sentence, [".", "?", "!", ","]))
```

```shell
$ Hi ,  my name is john .
```

&nbsp;</br>
**노이즈 유형 (2) 대소문자 : First, open the first chapter.**</br>
&nbsp;</br>
`First`와 `first`의 의미는 같지만, 컴퓨터가 인지할 때는 다른 단어로 인지한다. 이를 방지하기 위해 **모든 단어를 소문자로 바꾸는 방식**으로 노이즈를 제거한다.

```python
sentence = "First, open the first chapter."

print(sentence.lower())
```

```shell
$ first, open the first chapter.
```
&nbsp;</br>

```python
## lower()와 비슷한 기능을 하는 upper() 적용시켜보기
sentence = "First, open the first chapter."

# Q. sentence의 모든 단어를 대문자로 바꿔보세요. 
# 힌트: upper() 함수를 사용해 보세요!
print(sentence.upper())
```

```shell
FIRST, OPEN THE FIRST CHAPTER.
```

&nbsp;</br>
**노이즈 유형 (3) 특수문자**</br>
`ten-year-old`나 `seven-year-old` 같은 나이 표현들을 특수 유형으로 간주하여 처리하지 않으면 각각의 단어별로 토큰화가 되어버릴 수 있다. 이를 방지하기 위해 `정규 표현식(Regular expression; a.k.a regex)`를 사용하여 노이즈를 제거한다.

```python
import re

sentence = "He is a ten-year-old boy."
sentence = re.sub("([^a-zA-Z.,?!])", " ", sentence)

print(sentence)
```

```shell
He is a ten year old boy.
```

[Python 정규식 연산(Regex)](https://docs.python.org/ko/3/library/re.html)</br>


&nbsp;</br>
문장을 정제하는 함수 정의하고 확인하는 예시
```python
# From The Project Gutenberg
# (https://www.gutenberg.org/files/2397/2397-h/2397-h.htm)

corpus = \
"""
In the days that followed I learned to spell in this uncomprehending way a great many words, among them pin, hat, cup and a few verbs like sit, stand and walk. 
But my teacher had been with me several weeks before I understood that everything has a name.
One day, we walked down the path to the well-house, attracted by the fragrance of the honeysuckle with which it was covered. 
Some one was drawing water and my teacher placed my hand under the spout. 
As the cool stream gushed over one hand she spelled into the other the word water, first slowly, then rapidly. 
I stood still, my whole attention fixed upon the motions of her fingers. 
Suddenly I felt a misty consciousness as of something forgotten—a thrill of returning thought; and somehow the mystery of language was revealed to me. 
I knew then that "w-a-t-e-r" meant the wonderful cool something that was flowing over my hand. 
That living word awakened my soul, gave it light, hope, joy, set it free! 
There were barriers still, it is true, but barriers that could in time be swept away.
""" 

def cleaning_text(text, punc, regex):
    # 노이즈 유형 (1) 문장부호 공백추가
    for p in punc:
        text = text.replace(p, " " + p + " ")

    # 노이즈 유형 (2), (3) 소문자화 및 특수문자 제거
    text = re.sub(regex, " ", text).lower()

    return text

print(cleaning_text(corpus, [".", ",", "!", "?"], "([^a-zA-Z0-9.,?!\n])"))
```

```shell
$ in the days that followed i learned to spell in this uncomprehending way a great many words ,  among them pin ,  hat ,  cup and a few verbs like sit ,  stand and walk .  
but my teacher had been with me several weeks before i understood that everything has a name . 
one day ,  we walked down the path to the well house ,  attracted by the fragrance of the honeysuckle with which it was covered .  
some one was drawing water and my teacher placed my hand under the spout .  
as the cool stream gushed over one hand she spelled into the other the word water ,  first slowly ,  then rapidly .  
i stood still ,  my whole attention fixed upon the motions of her fingers .  
suddenly i felt a misty consciousness as of something forgotten a thrill of returning thought  and somehow the mystery of language was revealed to me .  
i knew then that  w a t e r  meant the wonderful cool something that was flowing over my hand .  
that living word awakened my soul ,  gave it light ,  hope ,  joy ,  set it free !  
there were barriers still ,  it is true ,  but barriers that could in time be swept away . 
```

---

## 1-3. 분산표현: 바나나와 사과의 관계를 어떻게 표현할까?