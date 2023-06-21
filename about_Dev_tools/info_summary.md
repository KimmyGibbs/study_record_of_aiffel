# L-2
**Git**</br>
개발을 진행하며 작성하는 소스코드의 업데이트 버전을 기록/관리할 수 있는 *소스코드 버전 관리 시스템*</br>

**Github**</br>
Git으로 관리하는 프로젝트를 호스팅하고 시공간 제약 없이 협업할 수 있는 온라인 서비스</br>

## 로컬 Git에 Github 계정 정보 등록하기
```shell
$ git config --global user.email "my-email@gmail.com"
$ git config --global user.name "my-username"
```
Git에 등록한 config의 정보를 모두 확인하고 싶을 경우 다음을 입력해보자.
```shell
$ git config -l
```

## 내 컴퓨터에 로컬 저장소 만들기
```shell
$ cd ~
$ cd user
$ mkdir workplace
```

## Git으로 버전 관리 시작하기
```shell
## 로컬 저장소로 이동
$ cd workplace

## 로컬 저장소에 git 설정
$ git init

## 폴더 경로 확인
$ ls -a
. .. .git
$ cd .git
$ ls
HEAD branches config description hooks info objects refs
```

## README.md 파일 생성하기
```shell
$ cd ~/user/workplace
$ echo "# first-repository" >> README.md
```
ReadMe 파일 확인하기
```shell
$ ls
README.md

$ cat README.md
# first-repository
```

## git으로 변화 확인 및 지금 버전에 도장찍기
Git의 추적내용을 확인하기 (status)
```shell
$ git status
On branch main

No commits yet

Untracked files:
    (use "git add <file>..." to include in what will be committed)
    README.md

nothing added to commit but untracked files present (use "git add" to track)
```

로컬 파일의 내역을 추적하기 위해 Git의 작업대(stage)에 추가(add)
```shell
$ git add README.md
On branch main

No commits yet

Changes to be committed:
    (use "git rm --cached <file>..." to unstage)
        new file: README.md
```

Git의 작업대에 기록중인 파일을 저장소(repository)에 확정하기 위해 명령(commit)어 수행
```shell
$ git commit -m "first commit"
[main (root-commit) kr98765] first commit
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

## Github에 저장소(repository) 만들기
- 참고 링크: https://aboneu.tistory.com/482

## 로컬 저장소와 원격 저장소 연결하기
```shell
$ cd ~/user/workplace
$ git remote add origin https://github.com/USERNAME/first-repository.git

## USERNAME의 경우 자신이 로그인해서 설정한 username이 들어간다.
```

## Github에서 토큰 생성하기
- 참고 링크: https://wooono.tistory.com/460

## 로컬 저장소의 기로을 원격저장소로 전송하기
```shell
$ git push origin main
Username for 'https://github.com': [계정에 사용된 이메일을 입력하세요]
Password for 'https://[계정에 사용된 이메일]@github.com': [비밀번호(토큰)을 입력하세요]
Enumerating objects: 3, done.
Counting objects: 100% (3/3), done.
Writing objects: 100% (3/3), 230 bytes | 230.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/USERNAME/first-repository.git
    * [new branch]  main -> main
```
Git 브랜치란? [Link](https://backlog.com/git-tutorial/kr/stepup/stepup1_1.html)

## 원격 저장소를 로컬로 가져오기
```shell
$ cd ~/user/workplace
$ git clone https://github.com/USERNAME/first-repository.git
Cloning into 'first-repository'...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 3 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
```
복사 확인 방법
```shell
## 저장소(repository)를 로컬에서 확인
$ ls
first-repository
## 로컬 타겟 디렉토리(first-repository)로 이동
$ cd first-repository
## 디렉토리 아래 파일 목록 조회
$ ls
README.md
## 파일의 내용 확인
$ cat README.md
# first-repository
```

## 로컬로 가져온 원격 저장소 다시 push해보기
```shell
## README 파일에 새로운 한 줄 추가하기
$ echo "add new contents" >> README.md
## 파일 내용 확인
$ cat README.md
# first-repository
add new contents
## Git 상태 체크하기 (status)
$ git status
On branch main
Changes not staged for commit:
    (use "git add <file>..." to update what will be commited)
    (use "git restore <file>..." to discard changes in working directory)
    modified: README.md

no changes added to commit (use "git add" and/or "git commit -a")
## ADD, COMMIT 하여 변경내용을 원격저장소에 기록하기
$ git add README.md
$ git commit -m "new contents"
[main k98765r] new contents
    1 file changed, 1 insertion(+)
## 확정(commit)된 코드를 원격저장소에 저장하기(push)
$ git push origin main
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Writing objects: 100% (3/3), 276 bytes | 276.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/USERNAME/first-repository.git
    123a45b..c98765d main -> main
```

## 로컬 저장소를 원격 저장소의 내용과 같게 업데이트하자!
```shell
## 로컬 저장소 파일 확인하기
$ cd ~/user/workplace
$ ls
README.md
$ cat README.md
# first-repository
## 로컬 저장소의 상태를 원격저장소의 상태와 동일하게 업데이트하기 (pull)
$ git pull origin main
remote: Enumerating objects: 5, done.
remote: Counting objects: 100% (5/5), done.
remote: Total 3 (delta 0), reused 3 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github/USERNAME/first-repository
    * branch    main -> FETCH_HEAD
    123a45b..c98765d main -> origin/main
Updatin 123a45b..c98765d
Fast-forward
    README.md | 1 +
    1 file changed, 1 insertion(+)
## 결과 확인하기
$ cat README.md
# first-repository
add new contents
```

## 마크다운 사용법
- 참고링크: [Link](https://gist.github.com/ihoneymon/652be052a0727ad59601)