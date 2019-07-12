# coding:utf-8
# Type: Public

import matplotlib.pyplot as plt
from matplotlib import animation
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue


class LineConfig(object):
    '''
    绘制在monitor上的线型config
    '''

    LineStyleSolid = "-"
    LineStyleDashed = "--"
    LineStyleDotted = ":"
    LineStyleDashDot = "-."

    LineMarkerPoint = "."
    LineMarkerTriangle = "^"
    LineMarkerSquare = "s"
    LineMarkerStar = "*"
    LineMarkerCross = "X"
    LineMarkerDiamond = "D"

    def __init__(self,linewidth=1.0,color=(0,0,0),line_style="-",line_marker="None"):
        self.linewidth = linewidth
        self.color = color
        self.line_style = line_style
        self.line_marker = line_marker


class RuntimeLineChart(object):
    '''

    在RL训练过程中实时监控参数的线图工具，使用字典传参，初始给出变量名-config的字典设置线型
    之后更新给出变量名-value的字典更新数据

    RL训练过程中的参数monitor，这个monitor是每个environment构建好之后需要debug的工具
    因为matplotlib只能够在主线程，因此多线程不能使用，这里用的是多进程
    使用多进程的Queue进行进程间数据传递
    注意Queue的get方法会阻塞直到另一个线程put内容到queue中！

    算法：调用这里面的update data函数，然后更新buffer的值用来显示
    如果buffer不更新就会使用上一次的值
    更新的interval可以设置
    '''

    def __init__(self,vname_to_line_config_dict=None,ylim=(0,70),window_area=50,update_interval=0.01):
        '''
        :param vname_to_line_config_dict:
        以变量名作为key，config数组作为value，
        :param ylim:
        :param xlim:
        :param window_area:   显示历史多少个数据，避免x无限大
        :param update_interval:
        '''
        self.y_buff = 10
        self.update_interval = update_interval
        self.ylim = ylim
        self.window_area = window_area
        self.queue = MPQueue()
        self.vname_to_lines = {}
        self.vname_to_line_config_dict = vname_to_line_config_dict
        self.vname_to_data = []

    def run(self):
        fig, ax = plt.subplots()
        for vname in self.vname_to_line_config_dict:
            config = self.vname_to_line_config_dict[vname]
            assert isinstance(config,LineConfig)
            line, = ax.plot([], [], lw=config.linewidth,color=config.color,marker=config.line_marker,
                            linestyle=config.line_style,label=vname)
            # 变量名到line对象，用于之后根据变量名到数值进行line更新
            self.vname_to_lines[vname] = line
        ax.set_ylim(*self.ylim)
        ax.grid()
        ax.legend()
        xdata = []
        self.vname_to_data = {}
        for key in self.vname_to_lines:
            self.vname_to_data[key] = []

        def run(data):
            t, y = data
            xdata.append(t)
            ax.set_xlim(t - self.window_area, t) # 曲线会随着移动，移动保持的历史数据量是time area
            ax.figure.canvas.draw()
            _return = []
            for vname in y:
                line = self.vname_to_lines[vname]
                data = y[vname]
                self.vname_to_data[vname].append(data)
                line.set_data(xdata, self.vname_to_data[vname])
                _return.append(line)
            return _return

        def __gen_data():
            x = []
            r = 0
            while 1:
                # 注意这里的data需要从函数中得到
                data = self.queue.get()  # 添加新的y,注意这个get方法是会阻塞这个线程来等待put的
                x.append(r)  # 时间
                r += 1
                yield float(x[-1]), data  # 每次yield列表的最后一个元素，考虑到数据最后要存储到文本中。

        # 会调用self.__gen_data,然后把结果传入run中
        ani = animation.FuncAnimation(fig, run, __gen_data, blit=True, interval=self.update_interval,
                                      repeat=False)
        plt.show()

    def update_data(self,new_y):
        self.queue.put(new_y)

    @staticmethod
    def test():
        import random

        vname_to_line_config = {
            "v1": LineConfig(color=(1, 0, 0)),
            "v2": LineConfig(),
            "v3": LineConfig(line_marker=LineConfig.LineMarkerDiamond, line_style=LineConfig.LineStyleDashed),
        }

        a = RuntimeLineChart(window_area=200, vname_to_line_config_dict=vname_to_line_config)
        p = Process(target=a.run)
        p.start()
        while 1:
            a.update_data({
                "v1": random.randint(0, 35),
                "v2": random.randint(0, 25),
                "v3": 45
            })
            pass

    @staticmethod
    def tst_multi_thread():
        import random
        class T():
            def __init__(self):
                # 这个是进程间通信用的Queue
                self.queue = MPQueue()
            def run(self):
                while 1:
                    a  = self.queue.get()
                    print(a)
                    if a:
                        print(a)
                    else:
                        print("none")
            def put_q(self,value):
                self.queue.put(value)


        t = T()
        p = Process(target=t.run)
        p.start()

        while 1:
            t.put_q(random.randint(0,70))
            pass


if __name__ == '__main__':
    #tst_multi_thread()
    RuntimeLineChart.test()





