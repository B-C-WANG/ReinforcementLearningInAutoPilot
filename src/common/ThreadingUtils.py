# coding:utf-8
import time
import threading


def invoke_repeating(start_delay, interval, func):
    time.sleep(start_delay)
    def run():
        while 1:
            time.sleep(interval)
            func()
    t = threading.Thread(target=run)
    t.start()



# 每次间隔执行某个函数
def invoke_repeating_old(start_delay, interval, func):
    '''
    开始执行的delay
    执行周期
    执行函数
    '''
    time.sleep(start_delay)
    def warp_():
        func()
        # 递归调用自己,实现循环
        timer = threading.Timer(interval,warp_)
        timer.start()
    threading.Timer(interval,warp_).start()


if __name__ == '__main__':
    index = 0
    def print_something():
        print(index)

    invoke_repeating(1,0.02,print_something)
    while 1:pass



