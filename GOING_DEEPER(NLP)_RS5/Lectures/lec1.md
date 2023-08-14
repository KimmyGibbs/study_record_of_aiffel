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
**단어의 희소 표현과 분산 표현**</br>
> 단어를 벡터로 표현하려고 하는 방법들

---

**희소 표현(Sparse representation)**</br>
- 단어를 고차원 벡터로 변환하는 방법
    - 고정된 크기의 벡터가 아니다
    - 이진화(binary) 또는 빈도수(frequency) 등의 방식으로 표현한다.
    - 단어의 존재 유무만 나타낸다.
        - 벡터 공간 상에서 거리 측정하기 어렵다.
        - 단어 간의 의미 관계를 파악하기 어렵다.

&nbsp;</br>
사람의 성별과 연령을 희소표현으로 나타낸 예시
> 위 경우에는 **적어도 2차원의 벡터가 필요함**
```json
{
    //     [성별, 연령]
    남자: [-1.0, 0.0], // 이를테면 0.0 이 "관계없음 또는 중립적" 을 의미할 수 있겠죠!
    여자: [1.0, 0.0],
    소년: [-1.0, -0.7],
    소녀: [1.0, -0.7],
    할머니: [1.0, 0.7],
    할아버지: [-1.0, 0.7],
    아저씨: [-1.0, 0.2],
    아줌마: [1.0, 0.2]
}
```

*희소표현의 문제점*</br>
단어의 속성이 커질수록 **너무 고차원의 벡터가 필요하다**</br>
*희소 표현의 워드 벡터끼리는 단어들간의 의미적 유사도를 계산할 수 없다.*</br>
&nbsp;</br>
> 두 고차원 벡터의 유사도는 **코사인 유사도(Cosine Similarity)**를 통해 구할 수 있다.
>> [Wikidocs: Cosine Similarity](https://wikidocs.net/24603)

---

**분산 표현(Distributed representation)**</br>
- Embedding 레이어를 사용하여 각 단어가 몇 차원의 속성을 가질지 정의하는 방식의 표현법
    - 단어르 고정된 크기의 벡터로 표현하는 방식
    - 하나의 단어를 여러 차원의 값으로 나타내는 방식
    - 단어 간의 거리를 측정
        - 단어 간 의미와 관련성을 파악할 수 있음

```python
# 100개의 단어를 256차원의 속성으로 표현하는 예시
embedding_layer = tf.keras.layers.Embedding(input_dim=100, output_dim=256)
```

---

## 1-4. 토큰화 : 그녀는? 그녀+는?
토큰화?
- 한 문장에서 단어의 수를 정의하는 기법

---

**공백 기반 토큰화**
- 자연어의 노이즈를 제거하는 방법 중 하나
    - `Hi,`를 `Hi`와 `,`로 나누기 위해 문장부호 양옆에 공백을 추가함

```python
## 공백 기반 토큰화 예시코드
corpus = \
"""
in the days that followed i learned to spell in this uncomprehending way a great many words ,  among them pin ,  hat ,  cup and a few verbs like sit ,  stand and walk .  
but my teacher had been with me several weeks before i understood that everything has a name . 
one day ,  we walked down the path to the well house ,  attracted by the fragrance of the honeysuckle with which it was covered .  
some one was drawing water and my teacher placed my hand under the spout .  
as the cool stream gushed over one hand she spelled into the other the word water ,  first slowly ,  then rapidly .  
i stood still ,  my whole attention fixed upon the motions of her fingers .  
suddenly i felt a misty consciousness as of something forgotten a thrill of returning thought  and somehow the mystery of language was revealed to me .  
i knew then that  w a t e r  meant the wonderful cool something that was flowing over my hand .  
that living word awakened my soul ,  gave it light ,  hope ,  joy ,  set it free !  
there were barriers still ,  it is true ,  but barriers that could in time be swept away . 
"""

tokens = corpus.split()

print("문장이 포함하는 Tokens:", tokens)
```
```shell
문장이 포함하는 Tokens: ['in', 'the', 'days', 'that', 'followed', 'i', 'learned', 'to', 'spell', 'in', 'this', 'uncomprehending', 'way', 'a', 'great', 'many', 'words', ',', 'among', 'them', 'pin', ',', 'hat', ',', 'cup', 'and', 'a', 'few', 'verbs', 'like', 'sit', ',', 'stand', 'and', 'walk', '.', 'but', 'my', 'teacher', 'had', 'been', 'with', 'me', 'several', 'weeks', 'before', 'i', 'understood', 'that', 'everything', 'has', 'a', 'name', '.', 'one', 'day', ',', 'we', 'walked', 'down', 'the', 'path', 'to', 'the', 'well', 'house', ',', 'attracted', 'by', 'the', 'fragrance', 'of', 'the', 'honeysuckle', 'with', 'which', 'it', 'was', 'covered', '.', 'some', 'one', 'was', 'drawing', 'water', 'and', 'my', 'teacher', 'placed', 'my', 'hand', 'under', 'the', 'spout', '.', 'as', 'the', 'cool', 'stream', 'gushed', 'over', 'one', 'hand', 'she', 'spelled', 'into', 'the', 'other', 'the', 'word', 'water', ',', 'first', 'slowly', ',', 'then', 'rapidly', '.', 'i', 'stood', 'still', ',', 'my', 'whole', 'attention', 'fixed', 'upon', 'the', 'motions', 'of', 'her', 'fingers', '.', 'suddenly', 'i', 'felt', 'a', 'misty', 'consciousness', 'as', 'of', 'something', 'forgotten', 'a', 'thrill', 'of', 'returning', 'thought', 'and', 'somehow', 'the', 'mystery', 'of', 'language', 'was', 'revealed', 'to', 'me', '.', 'i', 'knew', 'then', 'that', 'w', 'a', 't', 'e', 'r', 'meant', 'the', 'wonderful', 'cool', 'something', 'that', 'was', 'flowing', 'over', 'my', 'hand', '.', 'that', 'living', 'word', 'awakened', 'my', 'soul', ',', 'gave', 'it', 'light', ',', 'hope', ',', 'joy', ',', 'set', 'it', 'free', '!', 'there', 'were', 'barriers', 'still', ',', 'it', 'is', 'true', ',', 'but', 'barriers', 'that', 'could', 'in', 'time', 'be', 'swept', 'away', '.']
```

---

**형태소 기반 토큰화**
- 문장을 분리할 때, 공백이 아닌 **형태소**를 기준으로 분리(`Tokenizing`)하는 기법
> 형태소?
>> (명사) 뜻을 가진 가장 작은 말의 단위
>>> ex) `오늘도 공부만 한다` &rarr; `오늘`, `도`, `공부`, `만`, `한다`

한국어 형태소 분석기
- KoNLPy
    - [파이썬 한국어 NLP - KoNLPy 0.4.3 documentation](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)
- [한국어 형태소 분석기 성능 비교](https://iostream.tistory.com/144)

아래는 한국어 형태소 분석기를 사용하는 비교실험 예제</br>
1. konlpy 및 Mecab의 설치 여부 확인
```python
from konlpy.tag import Hannanum,Kkma,Komoran,Mecab,Okt
```
만약 `import` 에러가 발생한다면 아래의 과정을 수행하라고 함
```shell
$ pip install konlpy
$ cd ~/aiffel/text_preprocess
$ git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
$ cd Mecab-ko-for-Google-Colab
$ bash install_mecab-ko_on_colab190912.sh
```
위 과정 이후 `JVMNotFoundException`에러 발생 시 아래와 같이 자바 설치
```shell
$ sudo apt update
$ sudo apt install default-jre
```
2. 설치 완료 후 다음 문장을 어떻게 해석하는지 비교해 보자
```shell
코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤
전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.
```
3. 코드 예시
```python
tokenizer_list = [Hannanum(),Kkma(),Komoran(),Mecab(),Okt()]

kor_text = '코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤 전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.'

for tokenizer in tokenizer_list:
    print('[{}] \n{}'.format(tokenizer.__class__.__name__, tokenizer.pos(kor_text)))
```
```shell
Hannanum] 
[('코로나바이러스', 'N'), ('는', 'J'), ('2019년', 'N'), ('12월', 'N'), ('중국', 'N'), ('우한', 'N'), ('에서', 'J'), ('처음', 'M'), ('발생', 'N'), ('하', 'X'), ('ㄴ', 'E'), ('뒤', 'N'), ('전', 'N'), ('세계', 'N'), ('로', 'J'), ('확산', 'N'), ('되', 'X'), ('ㄴ', 'E'), (',', 'S'), ('새롭', 'P'), ('은', 'E'), ('유형', 'N'), ('의', 'J'), ('호흡기', 'N'), ('감염', 'N'), ('질환', 'N'), ('이', 'J'), ('ㅂ니다', 'E'), ('.', 'S')]
[Kkma] 
[('코로나', 'NNG'), ('바', 'NNG'), ('이러', 'MAG'), ('슬', 'VV'), ('는', 'ETD'), ('2019', 'NR'), ('년', 'NNM'), ('12', 'NR'), ('월', 'NNM'), ('중국', 'NNG'), ('우', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('에', 'VV'), ('서', 'ECD'), ('처음', 'NNG'), ('발생', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('뒤', 'NNG'), ('전', 'NNG'), ('세계', 'NNG'), ('로', 'JKM'), ('확산', 'NNG'), ('되', 'XSV'), ('ㄴ', 'ETD'), (',', 'SP'), ('새', 'NNG'), ('롭', 'XSA'), ('ㄴ', 'ETD'), ('유형', 'NNG'), ('의', 'JKG'), ('호흡기', 'NNG'), ('감염', 'NNG'), ('질환', 'NNG'), ('이', 'VCP'), ('ㅂ니다', 'EFN'), ('.', 'SF')]
[Komoran] 
[('코로나바이러스', 'NNP'), ('는', 'JX'), ('2019', 'SN'), ('년', 'NNB'), ('12월', 'NNP'), ('중국', 'NNP'), ('우', 'NNP'), ('한', 'NNP'), ('에서', 'JKB'), ('처음', 'NNG'), ('발생', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETM'), ('뒤', 'NNG'), ('전', 'MM'), ('세계로', 'NNP'), ('확산', 'NNG'), ('되', 'XSV'), ('ㄴ', 'ETM'), (',', 'SP'), ('새롭', 'VA'), ('ㄴ', 'ETM'), ('유형', 'NNP'), ('의', 'JKG'), ('호흡기', 'NNG'), ('감염', 'NNP'), ('질환', 'NNG'), ('이', 'VCP'), ('ㅂ니다', 'EF'), ('.', 'SF')]
[Mecab] 
[('코로나', 'NNP'), ('바이러스', 'NNG'), ('는', 'JX'), ('2019', 'SN'), ('년', 'NNBC'), ('12', 'SN'), ('월', 'NNBC'), ('중국', 'NNP'), ('우한', 'NNP'), ('에서', 'JKB'), ('처음', 'NNG'), ('발생', 'NNG'), ('한', 'XSV+ETM'), ('뒤', 'NNG'), ('전', 'NNG'), ('세계', 'NNG'), ('로', 'JKB'), ('확산', 'NNG'), ('된', 'XSV+ETM'), (',', 'SC'), ('새로운', 'VA+ETM'), ('유형', 'NNG'), ('의', 'JKG'), ('호흡기', 'NNG'), ('감염', 'NNG'), ('질환', 'NNG'), ('입니다', 'VCP+EF'), ('.', 'SF')]
[Okt] 
[('코로나바이러스', 'Noun'), ('는', 'Josa'), ('2019년', 'Number'), ('12월', 'Number'), ('중국', 'Noun'), ('우한', 'Noun'), ('에서', 'Josa'), ('처음', 'Noun'), ('발생', 'Noun'), ('한', 'Josa'), ('뒤', 'Noun'), ('전', 'Noun'), ('세계', 'Noun'), ('로', 'Josa'), ('확산', 'Noun'), ('된', 'Verb'), (',', 'Punctuation'), ('새로운', 'Adjective'), ('유형', 'Noun'), ('의', 'Josa'), ('호흡기', 'Noun'), ('감염', 'Noun'), ('질환', 'Noun'), ('입니다', 'Adjective'), ('.', 'Punctuation')]
```

---

**사전에 없는 단어의 문제**
- 공백 기반, 형태소 기반의 토큰화 기법들은 모두 **의미를 가지는 단위로 토큰을 생성**
    - 데이터에 포함되는 모든 단어를 처리할 수 없음
    - 자주 등장한 상위 N개의 단어만을 사용하여 토큰 생성
    - 나머지는 `<unk>`과 같은 **특수한 토큰(Unknown Token)으로 치환**
> 이러한 전처리는 종종 큰 문제를 야기함
>> 토큰화 예시
>>> 코로나바이러스는 2019년 12월 중국 우한에서 처음 발생한 뒤
전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다. 
>>> &rarr;
>>> `<unk>`는 2019년 12월 중국 `<unk>`에서 처음 발생한 뒤
전 세계로 확산된, 새로운 유형의 호흡기 감염 질환입니다.

위 문장을 영어로 번역 시, 핵심 단어인 `코로나바이러스`와 `우한`을 모르면 번역을 수행할 수 없다.</br>

이를 **OOV(Out-Of-Vocabulary)** 문제라고 한다.
- **새로 등장한(본 적 없는)단어에 대해 약한 모습**을 보인다. (ex: 번역이 어색함 등)

---

## 1-5. 토큰화 : 다른 방법들
**BPE(Byte Pair Encoding)**
- 데이터에서 **가장 많이 등장하는 바이트 쌍(Byte Pair)**을 새로운 단어로 치환하여 압축하는 기법
- 1994년 고안된 알고리즘
- 초기에는 데이터 압축을 위한 알고리즘
    - 예시는 아래와 같다.

```shell
# 가장 많이 등장한 바이트 쌍 "aa"를 "Z"로 치환합니다.
aaabdaaabac
→ 
# "aa" 총 두 개가 치환되어 4바이트를 2바이트로 압축하였습니다.
ZabdZabac

# 그다음 많이 등장한 바이트 쌍 "ab"를 "Y"로 치환합니다.
Z=aa
→ 
# "ab" 총 두 개가 치환되어 4바이트를 2바이트로 압축하였습니다.
ZYdZYac

# 여기서 작업을 멈추어도 되지만, 치환된 바이트에 대해서도 진행한다면
Z=aa
# 가장 많이 등장한 바이트 쌍 "ZY"를 "X"로 치환합니다.
Y=ab
→ 
XdXac
Z=aa
Y=ab
# 압축이 완료되었습니다!
X=ZY
```

해당 알고리즘을 토큰화에 적용하자고 제안(2015년)
1. 모든 단어를 문자(바이트)들의 집합으로 취급
2. 자주 등장하는 문자 쌍을 합침
3. 접두사 / 접미어의 의미를 캐치할 수 있다.
4. 처음 등장하는 단어는 문자(알파벳)들의 조합으로 표시

**OOV 문제를 완전히 해결**할 수 있다.</br>
> [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)

아래는 Python 예제
```python
import re, collections

# 임의의 데이터에 포함된 단어들입니다.
# 우측의 정수는 임의의 데이터에 해당 단어가 포함된 빈도수입니다.
vocab = {
    'l o w '      : 5,
    'l o w e r '  : 2,
    'n e w e s t ': 6,
    'w i d e s t ': 3
}

num_merges = 5

def get_stats(vocab):
    """
    단어 사전을 불러와
    단어는 공백 단위로 쪼개어 문자 list를 만들고
    빈도수와 쌍을 이루게 합니다. (symbols)
    """
    pairs = collections.defaultdict(int)
    
    for word, freq in vocab.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):             # 모든 symbols를 확인하여 
            pairs[symbols[i], symbols[i + 1]] += freq  # 문자 쌍의 빈도수를 저장합니다. 
        
    return pairs

def merge_vocab(pair, v_in):
    """
    문자 쌍(pair)과 단어 리스트(v_in)를 입력받아
    각각의 단어에서 등장하는 문자 쌍을 치환합니다.
    (하나의 글자처럼 취급)
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
        
    return v_out, pair[0] + pair[1]

token_vocab = []

for i in range(num_merges):
    print(">> Step {0}".format(i + 1))
    
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)  # 가장 많은 빈도수를 가진 문자 쌍을 반환합니다.
    vocab, merge_tok = merge_vocab(best, vocab)
    print("다음 문자 쌍을 치환:", merge_tok)
    print("변환된 Vocab:\n", vocab, "\n")
    
    token_vocab.append(merge_tok)
    
print("Merged Vocab:", token_vocab)
```
결과 예시
```shell
>> Step 1
다음 문자 쌍을 치환: es
변환된 Vocab:
 {'l o w ': 5, 'l o w e r ': 2, 'n e w es t ': 6, 'w i d es t ': 3} 

>> Step 2
다음 문자 쌍을 치환: est
변환된 Vocab:
 {'l o w ': 5, 'l o w e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> Step 3
다음 문자 쌍을 치환: lo
변환된 Vocab:
 {'lo w ': 5, 'lo w e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> Step 4
다음 문자 쌍을 치환: low
변환된 Vocab:
 {'low ': 5, 'low e r ': 2, 'n e w est ': 6, 'w i d est ': 3} 

>> Step 5
다음 문자 쌍을 치환: ne
변환된 Vocab:
 {'low ': 5, 'low e r ': 2, 'ne w est ': 6, 'w i d est ': 3} 

Merged Vocab: ['es', 'est', 'lo', 'low', 'ne']
```

---

**WPM(Wordpiece Model)**
- 하나의 단어를 여러개의 sub-word 집합으로 보는 방법
> 예시) `preview`, `predict`의 경우
>> 두 단어를 별개로 생각하지 않고 `pre + view`, `pre + dict`로 보면서, 접두어인 `pre`의 의미를 고려하여 토큰화를 하면 학습률이 올라갈 것이다라는 접근법
    - OOV(Out-Of-Vocabulary) 문제를 해결하기 위해 고안된 기법

*추가 내용*
- 구글에서 BPE을 변형해 제안한 알고리즘
- WPM은 BPE에 대해 **두 가지 차별성**을 가짐
    1. 공백 복원을 위해 단어 시작부에 `_`를 추가
    2. 빈도수 기반이 아닌 가능도(Likelihood)를 증가시키는 방향으로 문자 쌍을 합침
        - 더 `그럴듯한` 토큰 생성

> 예시구문) i am a boy and you are a girl
>> [`_i`, `_am`, `_a`, `_b`, `o`, `y`, `_a`, `n`, `d`, `_you`, `_a`, `_gir`, `l`] 로 토큰화
>>> 문장 복원 시 **1) 모든 토큰 합친 후**, **2) `_`를 공백으로 치환**

[Paper: Japanese and Korean voice search](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf)</br>
[확률과 가능도(likelihood) 그리고 최대우도추정](https://jjangjjong.tistory.com/41)</br>
[SentencePiece-google's tokenizer](https://github.com/google/sentencepiece)</br>
[한국어를 위한 토크나이저 - soynlp](https://github.com/lovit/soynlp)</br>

---

## 1-6. 토큰에게 의미를 부여하기


## 1-7. 마무리하며