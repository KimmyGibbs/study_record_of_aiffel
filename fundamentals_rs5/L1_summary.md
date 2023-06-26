# 함수와 변수
함수: 불려진 시점에 특정한 작업을 수행; 입력값과 출력값은 경우에 따라 존재하거나 존재하지 않음
```python
# 출력함수
print('Hello)
# 값의 타입 체크
type('Hello')
# 문자열 합치기
print('Hello ' + 'world!')
```
변수: 값이 저장된 곳. 함수의 입력값 또는 출력값에 사용될 수 있다.

# 제어문
IF: 명제가 참인지 거짓인지에 따라 코드 실행여부를 결정할 수 있다.
&nbsp;</br>
WHILE: 조건이 참인 경우 반복 실행
&nbsp;<br/>
FOR: 특정 목록에서 값을 하나씩 꺼내서 반복문을 실행
```python
for hangul in '김민식':
    print(hangul)
```

# 자료형
변수의 성질. 값들의 종류 하나하나를 자료형(data type)
```python
print(type(1))
```
## 정수
integer, 'int'
&nbsp;</br>
'int'는 양의 정수와 음의 정수를 모두 포함함.
## 부동소수점
floating-point number, 'float'
&nbsp;</br>
```python
print(type(1.1))
```
### 32bit 고정소수점, 부동소수점
고정소수점: 1bit(부호) + 16bit(정수) + 15bit(소수)
&nbsp;</br>
부동소수점: 1bit(부호) + 8bit(지수) + 23bit(가수)
```text
십진수 1.0을 고정소수점과 부동소수점으로 각각 표현해보기

// 고정소수점
1. 부호체크 (양수)
2. 정수체크 (1) --> 16 bit 맞추어 값 넣기(padding)
3. 소수체크 (0) --> 15 bit 맞추거 값 넣기(padding)

>> 0 0000 0000 0000 0001 0000 0000 0000 000

// 부동소수점
1. 부호체크 (양수)
2. 십진수를 이진수로 변경 (1은 1)
3. 2번의 값을 2의 지수형태로 표현 (1 = 2^0)
4. 지수부 세팅 (0)
5. IEEE 규칙을 참고하기
 * 8 bit 케이스에서 지수의 범위는 -127 ~ 128이며 표현은 0000 0000 ~ 1111 1111 로 한다.
6. 4번에서 구한 값을 IEEE 규정에 맟추어 8 bit로 변환하기
7. 지수 0은 IEEE 8 bit로 0111 1111

>> 0 0111 1111 0000 0000 0000 0000 0000 000
```
## NoneType
python에서 값이 없을을 뜻함
&nbsp;</br>
타 언어에서는 주로 `null`로 표현한다.
## Boolean
참 또는 거짓의 논리값을 가지는 자료형
## 문자열
string 타입, str로 표현
### 컨테이너 자료형
python에서 주로 사용하는 용어
&nbsp;</br>
반복 가능한 객체 중에서 유한한 길이를 가지는 자료형들을 모두 컨테이너 자료형이라고 한다.
```python
## 특정 인덱스의 값을 출력
message = 'Hello'
print(message[1])
# e

## 특정 인덱스의 값을 출력 (뒤에서부터)
message = 'Hello'
print(message[-1])
# o

## 특정 문자열만을 가져오는 slice
message = 'Hello'
print(message[0:4])
# Hell
## slice 추가에시 1
message = 'Hello'
print(message[2:])
print(message[:4])
# llo
# Hell
## slice 추가예시 2
message = 'Hello'
print(message[::1])  # 기본 (한 칸 씩)
print(message[:-1:1])  # 기본 (한 칸 씩)
print(message[::2])  # 두 칸 씩
# Hello
# Hell
# Hlo

## 값을 거꾸로 출력(reverse)
message = 'Hello'
print(message[::-1])  # 거꾸로 출력
# olleH
```
## 튜플; tuple()
여러 값들을 괄호(`()`)로 묶어 표현하는 자료형
```python
numbers = (1, 2, 3)
mixed_tuple = ('Hello', 0, True)
# 튜플에는 어떤 값이든 들어갈 수 있다.
```
문자열과 같이 `tuple[index]` 문법을 사용하여 특정 위치의 값을 읽을 수 있다.
&nbsp;</br>
하지만 **값을 변경할 수 없다**.
## 리스트; list()
튜플과 비슷하지만 여러 값들을 대괄호(`[]`)로 묶어 표현한다
&nbsp;</br>
튜플과는 다르게 리스트 내부의 값들을 변경할 수 있다.
## 딕셔너리; dict()
여러 값들을 `key-value` 페어 형태로 만든 후 중괄호(`{}`)로 묶어 표현하는 자료형
```python
conductor = {'first_name': '단테', 'last_name': '안'}
print(conductor['first_name'])
```