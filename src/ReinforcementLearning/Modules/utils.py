# coding:utf-8
# Type: Public
import numpy as np
import warnings
import math

class MathUtils(object):
    @staticmethod
    def angle_of_two_unit_vector_use_cos(vector1,vector2):
        cos_angle = abs(vector1.dot(vector2))
        angle = np.arccos(cos_angle)
        if np.isnan(angle):
            return 0
        # 在一二象限,根据相对位置得到正负
        if (vector1[1]<vector2[1]):
            angle =  -angle
        if (vector1[0]<vector2[0]):
            angle = -angle
        return angle


    @staticmethod
    def angle_of_two_unit_vector_use_sin(vector1, vector2):
        cos_angle = vector1.dot(vector2)
        return np.arcsin(cos_angle)


    @staticmethod
    def calc_angle(x1, y1, x2, y2):
        angle = 0
        dy = y2 - y1
        dx = x2 - x1
        if dx == 0 and dy > 0:
            angle = 0
        if dx == 0 and dy < 0:
            angle = 180
        if dy == 0 and dx > 0:
            angle = 90
        if dy == 0 and dx < 0:
            angle = 270
        if dx > 0 and dy > 0:
            angle = math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy > 0:
            angle = 360 + math.atan(dx / dy) * 180 / math.pi
        elif dx < 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        elif dx > 0 and dy < 0:
            angle = 180 + math.atan(dx / dy) * 180 / math.pi
        return angle

class LinearEpsilonAnnealingExplorer():
    """
    Exploration policy using Linear Epsilon Greedy
    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end

    线性收敛的e-greedy
    e-greedy指的是以一定概率e进行贪心，如果贪心就选择估值网络预测出来的最大的action，否则就完全随机选择！
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.
        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step
        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore
        Attributes:
            step (int) : Current step
        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)


class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis
        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history
        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0
        """
        self._buffer.fill(0)


class NoBufferMemory:
    def __init__(self):

        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class ReplayMemory(object):
    """
    Replay存储(前一状态st，动作a，下一状态st+1，奖励r，以及是否结束done)即(s,a,s',r,d)，简称为sdsrd
    每个Replay可以用来训练
    """

    def __init__(self, size, sample_shape, history_length=4):
        # 当前插入记录的位置，每次插入+1
        self._pos = 0
        # 记录总数
        self._count = 0
        # 最大存储空间
        self._max_size = size
        # 每次拿出来训练的一个样本的大小
        self._history_length = max(1, history_length)
        # 样本shape，相当于state的向量长度
        self._state_shape = sample_shape
        # state的矩阵形状，初始化就是不可变数组
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        # action的index，没有one hot编码
        self._actions = np.zeros(size, dtype=np.uint8)
        # reward
        self._rewards = np.zeros(size, dtype=np.float32)
        # 是否结束的bool值
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.
        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        # 每次按照顺序加入这些值
        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.
        Attributes:
            size (int): The minibatch size
        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        # 采样出所有的index
        while len(indexes) < size:
            index = np.random.randint(history_len, count)
            # 保证不重复
            if index not in indexes:


                # 需要保证index能够取出history len长度的(s,a,s',r,d),sasrd，所以index只能在history len ~ pos - history len之间
                # 注意这里是pos而不是count，因为数据只在pos有效，超出pos为空！
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.
        Attributes:
            size (int): Minibatch size
        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """

        # 获得起始的index
        indexes = self.sample(size)
        # 每次需要获得前后两个滑动窗口状态，作为s和st+1
        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)

        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.
        Attributes:
            index (int): State's index
        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """

        # 从state的index开始获得history length数目的state
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)



# 用于记录和打印消息
def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    """Helper function to store score and print statistics.

    Arguments:
      episode: Current episode
      episode_reward: Reward accumulated over the current episode
      worker_idx: Which thread (worker)
      global_ep_reward: The moving average of the global reward
      result_queue: Queue storing the moving average of the scores
      total_loss: The total loss accumualted over the current episode
      num_steps: The number of steps the episode took to complete
    """
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        # 如果到了第十步以后,就是前10步的滑动平均,小于10步就是前n步的滑动平均
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    # print(
    #     f"Episode: {episode} | "
    #     f"Moving Average Reward: {int(global_ep_reward)} | "
    #     f"Episode Reward: {int(episode_reward)} | "
    #     f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
    #     f"Steps: {num_steps} | "
    #     f"Worker: {worker_idx}"
    # )
    result_queue.put(global_ep_reward)
    return global_ep_reward


def colored_reward(reward,thushold=0):
    '''
    rewards大于阈值,显示绿色,小于则是黄色
    '''
    if reward > thushold:
        return "\033[0;32;40m" + str(reward) + "\033[0m"
    else:
        return "\033[0;33;40m" + str(reward) + "\033[0m"



