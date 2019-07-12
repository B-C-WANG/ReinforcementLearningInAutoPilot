# coding:utf-8
# Type: Public

import os
import threading
'''
使用docker启动carla运行环境
需要注意，docker默认启动的是town3的map！

docker启动命令：
sudo service docker start
sudo docker run -p 2000-2002:2000-2002 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.5
# 上面的port从2000到2002是需要的，实测如果只设置2000是链接不了太多的client的！
# 冒号前面的是本机端口，后面的是docker内部转发过去的端口，可以不用变，因为carla只监听2000端口！

如果sudo遇到问题，可以开启免密码：
sudo: no tty present and no askpass program specified
由于帐号并没有开启免密码导致的 
假设当前帐号为abc
切换到root下  
1    打开sudoers
vi /etc/sudoers
2    添加免密码
abc ALL = NOPASSWD: ALL



'''

def shut_down_docker():
    os.system("sudo service docker stop")



def launch_many_carla_docker(*port_list):
    base_command = "sudo service docker start;sudo docker run -p %s-%s:%s-%s --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 carlasim/carla:0.9.5 /bin/bash CarlaUE4.sh -world-port=%s; exec bash "
    command_string = "gnome-terminal "

    for port in port_list:
        # 前面的参数是本机的端口，会被映射到docker中的2000-2005端口，2000被carla监听！
        command_string  = command_string + " " + "--tab -e" + " \' bash -c \" " + base_command%(port,port+5,port,port+5,port) + "\"" +"\'"

    print(command_string)
    os.system(command_string)


    while 1:pass

if __name__ == '__main__':
    # 启动两个server，分别监听端口2000和2006，注意端口间隔6
    #launch_many_carla_docker(2000,2006)


    launch_many_carla_docker(2000)