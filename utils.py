import time

def execution_time(func):
    '''
    Декоратор для измерения времени выполнения функции
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        seconds = time.time() - start

        print(f'Execution time: {seconds // 60} min. {round(seconds % 60, 3)} sec.')
        return result

    return wrapper

def singleton(cls):
    '''
    Декоратор класса для реализации паттерна проектирования "Одиночка" (англ. "Singleton")
    '''

    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()

        return instances[cls]

    return getinstance