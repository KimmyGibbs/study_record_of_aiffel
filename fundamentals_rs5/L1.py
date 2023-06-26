# 피보나치 수열 로직 구현한 함수
def FiboOrigin(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return FiboOrigin(n-1) + FiboOrigin(n-2)
    

"""
피보나치킨 알고리즘 참고 링크
https://your-calculator.com/life/food/fibonacci-chicken
"""
# 2023.06.26 T 17:36 기준 수정이 필요한 코드
def FibonaChicken(n):
    init_res = FiboOrigin(n-1)
    #
    if init_res == 0:
        init_res = 1

    mod_res = n % init_res
    if mod_res == 0:
        return init_res
    else:
        return FiboOrigin(mod_res)

# 사람수 N명일때의 피보나치킨 수
N = 3
FibonaChicken(N)


# 피보나치 수열 경량화 (dict 자료형을 사용하여)
memory = {1: 1, 2: 1} 

def fibonacci(n):
    if n in memory:
        number = memory[n]
    else:
        number = fibonacci(n-1) + fibonacci(n-2)
        memory[n] = number
    return number