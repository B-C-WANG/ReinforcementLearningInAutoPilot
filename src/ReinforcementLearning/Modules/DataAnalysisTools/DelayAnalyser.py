# coding:utf-8
# Type: Public
import time


class DelayAnalyser(object):
    '''
    延迟分析工具
    '''
    def __init__(self):
        self.time_dict  ={}
        self.all_scope = set()

    def start(self,scope_name,offset=0):
        self.time_dict[scope_name+"start"] = time.time() + offset
        self.all_scope.update([scope_name])

    def end(self,scope_name,offset=0):
        self.time_dict[scope_name+"end"] = time.time() + offset

    def report_delay(self,scope_name=None):
        if scope_name is not None:
            try:
                print("Scope %s runned %s ms"%(scope_name,(self.time_dict[scope_name+"end"]-self.time_dict[scope_name+"start"])*1000))
            except KeyError:
                print("No such scope %s"%scope_name)
            return
        for scope_name in self.all_scope:
            try:
                print("Scope %s runned %s ms" % (
                scope_name, (self.time_dict[scope_name + "end"] - self.time_dict[scope_name + "start"]) * 1000))
            except KeyError:
                pass

    @staticmethod
    def test_delay_analysis():
        da = DelayAnalyser()
        da.start("test")
        time.sleep(0.1)
        da.end("test")
        da.start("test2")
        time.sleep(0.01)
        da.end("test2")
        da.report_delay("test2")
        da.report_delay()


if __name__ == '__main__':
    DelayAnalyser.test_delay_analysis()

