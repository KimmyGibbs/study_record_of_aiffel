# L-3
## 현재 컴퓨터의 유저 이름 표시하기
```shell
$ whoami
```
---
## 내 현재 경로 알아보기
```shell
$ pwd
/Users/mskim/Desktop/study_record_of_aiffel
```
---
## 현재 디렉토리 내의 모든 파일 목록 보기
```shell
$ ls
L-3_summary.md          README.md               onboarding_quests
```
목록을 자세히 봅시다
```shell
$ ls -al
total 16
drwxr-xr-x   6 mskim  staff  192  6 23 14:33 .
drwx------+ 11 mskim  staff  352  6 22 11:46 ..
drwxr-xr-x  14 mskim  staff  448  6 23 11:41 .git
-rw-r--r--   1 mskim  staff  364  6 23 14:37 L-3_summary.md
-rw-r--r--   1 mskim  staff   89  6 22 14:38 README.md
drwxr-xr-x   5 mskim  staff  160  6 23 11:41 onboarding_quests
```
---
## 원하는 디렉토리로 이동
```shell
$ cd onboarding_quests
$ pwd

/Users/mskim/Desktop/study_record_of_aiffel/onboarding_quests
```
## 상위 폴더로 올라가기
```shell
$ cd ..
$ pwd

/Users/mskim/Desktop/study_record_of_aiffel
```
## Home 디렉토리(세부경로의 맨 처음위치)로 이동하기
```shell
$ cd ~
$ pwd

/Users/mskim
```
---
## Ubuntu 20.04 환경에서 Anaconda 설치하기
```shell
## wget을 사용하는 방법
## 아나콘다를 설치할 디렉토리에서 명령어를 수행한다고 가정함
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh


## 공식 홈페이지에서 리눅스용 설치파일을 우분투에 다운로드 한 경우
## 다운로드한 파일(.sh 확장자)이 있는 위치에서 명령어를 실행한다고 가정함
$ bash Anaconda3-2019.10-Linux-x86_64.sh
```
### 이 외의 기타 설정 등의 세부사항 참고 링크 : [Here](https://mishuni.tistory.com/118)
---
## Anaconda를 활용한 가상환경 만들기
```shell
## 가상환경명을 'test_my_conda_env', 파이썬 버전을 3.9.7로 설정하여 생성
$ conda create -n test_my_conda_env python=3.9.7
```
---
## Anaconda 가상환경 리스트 확인하기
```shell
$ conda env list
```
---
## Anaconda 가상환경 실행하기
```shell
$ conda activate test_my_conda_env
```
### 실행이 완료되면 다음과 같은 형태의 명령어 상태가 확인됨
```shell
(test_my_conda_env) root@root_dir:~#
```
---
## Anaconda 가상환경 초기화 (Cloud on LMS)
```shell
# conda activate가 제대로 수행되지 않는 경우 다음과 같은 패턴의 에러 발생

CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

$ conda init <SHEEL_NAME>

Currently supproted shells are:
    - bash
    - fish
    - ...

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'
```
---
## 가상환경 종료
```shell
$ conda deactivate
```
---
## 가상환경 삭제
```shell
# 가상환경을 잘못 만들었을 경우 다음 명령어를 통해 가상환경 삭제
$ conda env remove -n test_my_conda_env
```