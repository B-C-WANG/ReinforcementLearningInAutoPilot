# coding:utf-8
# Type: MultiAuthor
import tensorflow as tf
import warnings
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import time
import traceback


# TODO:将模型中的小dense网络等公用


class IModel(object):

    def build(self):
        pass


class TFSaveLoadComponent(object):
    '''
    继承这个组件实现tf模型存储读取(使用python多继承)
    '''

    def save(self, global_step=None):
        filename = self.save_dir + "weights.ckpt"
        if global_step is None:
            self.saver.save(self.sess, filename)
        else:
            self.saver.save(self.sess, filename, global_step=global_step)
        print("Save weights Success, which has total step %s" % global_step)

    def load(self):
        try:
            model_file = tf.train.latest_checkpoint(self.save_dir)
            self.saver.restore(self.sess, model_file)
            print("Load from file %s Success" % model_file)
            time.sleep(1)
        except:
            traceback.print_exc()
            print("Can not load weights from file!")
            time.sleep(1)


class ConvFeatureConcatModel_v1(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = False


    '''
    TODO:
    尚未使用
    
    从image的卷积结果以及当前速度和转向的feature concat作为最终feature,Dense输出
    
    Debug记录: 
    报错UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 32: ordinal not in range(128)
    实际上是有其他的错误,但是在显示这个其他错误的时候报错工具出现了错误,
    所以为了显示实际的错误,需要修复这个bug
    为什么发生这个错误:因为文件路径中有中文,显示bug需要打印出对应的文件路径,所以有了bug
    解决方法:修改源代码更改编码格式等,在ide中修改没有权限,可以sudo  gedit error_interpolation.py
    在future import后面加上
    import sys
    reload(sys)
    sys.setdefaultencoding("utf-8")
    即可
    '''

    def __init__(self, image_input_shape=(64, 64, 3), additional_feature_shape=4, output_shape=5, gamma=0.99):
        self.gamma = gamma
        self.image_input_shape = image_input_shape
        assert len(image_input_shape) == 3, "Input shape should be (height,width,n_channel)"
        self.additional_feature_shape = additional_feature_shape
        self.output_shape = output_shape
        self.sess = tf.Session()
        self.build()

    def train(self):
        pass

    def predict_q(self):
        pass

    def build(self):
        '''
        1. 建立从图像和额外feature的state到q值估计的计算图
        2. 在1的基础上延伸模型,添加从pre states,rewards等等到loss的训练时使用的计算图
        使用时只需要得到目标output的graph,然后feed dict设置成相应输入即可
        所有input开头的都是feed dict可以添加的,所有output开头的都是模型的终点!
        '''

        # 建立state->各个action q值的模型
        # 图像输入
        self.input_image = tf.placeholder(tf.float32, shape=(
            None, self.image_input_shape[0], self.image_input_shape[1], self.image_input_shape[2]))
        # 额外的特征向量
        self.input_additional_feature = tf.placeholder(tf.float32, shape=(None, self.additional_feature_shape))
        self.input_true_y = tf.placeholder(tf.float32, shape=(None, self.output_shape))

        image_x = tf.keras.layers.Conv2D(kernel_size=(8, 8), filters=16, strides=4, activation="relu")(self.input_image)
        image_x = tf.keras.layers.Conv2D(kernel_size=(4, 4), filters=32, strides=2, activation="relu")(self.input_image)
        image_x = tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32, strides=1, activation="relu")(self.input_image)

        self.image_feature = tf.layers.Flatten()(image_x)

        self.total_feature = tf.concat([self.image_feature, self.input_additional_feature], axis=1)

        dense = tf.keras.layers.Dense(256)(self.total_feature)
        self.output_pred_y = tf.keras.layers.Dense(self.output_shape)(dense)

        '''
        这里的计算:
        我们希望当is done为1时使用A的值,当is done为0使用B的值:
        算法:
        假设is done为1010向量
        is done直接乘以A,得到A0A0
        is done减去1,取绝对值,得到0101,乘以B得到0B0B
        A0A0+0B0B等于ABAB
        '''
        # 因为后面涉及到float乘法,这里先设置为float
        self.input_is_done = tf.placeholder(tf.float32, shape=(None, 1))
        self.input_rewards = tf.placeholder(tf.float32, shape=(None, 1))
        self.__q_value = self.gamma * tf.reduce_max(
            # 采用最大的预测出来的q值
            self.output_pred_y, axis=1) + self.input_rewards

        self.__q_value = tf.reshape(self.__q_value, shape=(-1, 1))
        a_reward = self.input_is_done * self.input_rewards
        b_reward = tf.abs(self.input_is_done - 1) * self.__q_value

        self.q_targets = a_reward + b_reward

        print(self.q_targets.shape)

        # 虽然是onehot,但是之后需要涉及到浮点数乘法,因此需要设置成相应type
        self.input_actions_1_hot = tf.placeholder(tf.float32, shape=(None, self.output_shape))
        # 因为action是one hot的,所以直接相乘就是选择对应action下的q值
        self.q_acted = tf.reduce_sum(self.output_pred_y * self.input_actions_1_hot, axis=1)
        self.q_acted = tf.reshape(self.q_acted, shape=(-1, 1))

        self.loss = tf.losses.huber_loss(self.q_targets, self.q_acted)

        # 优化算法和优化操作
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=0.00025).minimize(self.loss)

    def show_graph_shape(self):
        # 用于检查各个层的shape是否是预期的,因为tf是计算图,计算前已经知道所有输出的shape了
        print("Input Image shape")
        print(self.input_image.shape)
        print("Input true y shape")
        print(self.input_true_y.shape)
        print("Image feature shape")
        print(self.image_feature.shape)
        print("Total Feature shape")
        print(self.total_feature.shape)
        print("Output y shape")
        print(self.output_pred_y.shape)
        print("Is done and rewards shape")
        print(self.input_is_done.shape)
        print(self.input_rewards.shape)
        print("Q targets shape")
        print(self.q_targets.shape)
        print("Q acted shape")
        print(self.q_acted.shape)
        print("loss shape")
        print(self.loss.shape)


class DenseModel_v1(IModel):
    Author = "BaoChuan Wang"
    AllowImport = True

    """
    QEval model是估值模型,从state给出各个action的Q值,用于进行下一步的action
    QEval model的具体参数在DQNAgent中给出,通过action和state等
    
    关于DQN模型
    模型在训练过程要用到两次!
    第一次是从post states直接预测q targets,第二是从pre states预测q acted然后求loss优化
    目前没有设计到将这一个模型共享到两次的使用中
    因此是:
    1.先从post states预测q targets
    2.然后用pre states预测q acted,作出误差训练
    也就是说训练只保留第二次的计算图进行优化
    """

    def __init__(self, input_space_size,
                 output_space_size, ckpt_directory,
                 hidden_size=(32, 16), gamma=0.9, model_save_step_interval=1000):
        assert (isinstance(input_space_size, tuple) and len(
            input_space_size) >= 1), "Should be tuple with a number, like (4,)"
        assert (isinstance(output_space_size, tuple) and len(
            output_space_size) >= 1), "Should be tuple with a number, like (4,)"
        self.input_space_size = input_space_size[0]
        self.output_space_size = output_space_size[0]
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.sess = tf.Session()
        self.build()
        self.tf_weights_inited = False
        self.n_trained_steps = 0
        self.save_step_interval = model_save_step_interval
        self.ckpt_directory = ckpt_directory

    def build(self):
        self.input_state = tf.placeholder(tf.float32, shape=(None, self.input_space_size), name="input_state")
        # self.input_true_y = tf.placeholder(tf.float32,shape=(None,self.output_shape))

        t = self.input_state
        for size in self.hidden_size:
            t = tf.keras.layers.Dense(units=size, activation=tf.keras.activations.relu)(t)
        self.output_pred_y = tf.keras.layers.Dense(units=self.output_space_size)(t)

        self.input_is_done = tf.placeholder(tf.float32, shape=(None, 1))
        self.input_rewards = tf.placeholder(tf.float32, shape=(None, 1), name="input_is_done")

        # 这里的问题在于,这个self.output_pred_y和下一个q acted的self.output_pred_y是不同的输入!

        # 求和之后注意和input rewards和shape一致
        self.__q_value = tf.reshape(self.gamma * tf.reduce_max(
            self.output_pred_y, axis=1), shape=(-1, 1)) + self.input_rewards

        # self.__q_value = tf.reshape(self.__q_value, shape=(-1, 1))

        a_reward = self.input_is_done * self.input_rewards
        b_reward = tf.abs(self.input_is_done - 1) * self.__q_value

        self.q_targets = a_reward + b_reward

        # 下面就是另一个模型,将q targets的计算和下面的训练分开
        self.input_actions_1_hot = tf.placeholder(tf.float32, shape=(None, self.output_space_size))
        # 这里用placeholder而不是使用计算图,将两个output pred y分开
        self.q_targets_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
        # self.q_acted = tf.reduce_sum(self.output_pred_y * self.input_actions_1_hot, axis=1)
        self.q_acted = tf.reduce_sum(self.output_pred_y * self.input_actions_1_hot, axis=1)
        self.q_acted = tf.reshape(self.q_acted, shape=(-1, 1))
        self.loss = tf.losses.huber_loss(self.q_targets_placeholder, self.q_acted)
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=0.00025).minimize(self.loss)
        self.n_trained_steps_tf = tf.Variable(0, tf.int32)
        self.saver = tf.train.Saver(max_to_keep=1)

    def predict_q(self, state):
        return self.sess.run(self.output_pred_y, feed_dict={
            self.input_state: state
        })

    def init_tf_weights(self):

        self.sess.run(tf.global_variables_initializer())
        self.tf_weights_inited = True

    def load_tf_weights(self):

        dir_ = self.ckpt_directory
        model_file = tf.train.latest_checkpoint(dir_)
        if model_file is None:
            warnings.warn("Can not load weights from file!")
        else:
            self.saver.restore(self.sess, model_file)
        # 读取训练了多少步骤的信息
        self.n_trained_steps = self.sess.run(self.n_trained_steps_tf)
        print("Load from file %s, which has step %s" % (model_file, self.n_trained_steps))

    def save_tf_weights(self):
        filename = self.ckpt_directory + "weights.ckpt"
        # 保存训练了多少步骤的信息
        self.sess.run(tf.assign(self.n_trained_steps_tf, self.n_trained_steps))
        self.saver.save(self.sess, filename, global_step=self.n_trained_steps)
        print("Save to file %s, which has step %s" % (filename, self.sess.run(self.n_trained_steps_tf)))

    def train(self, pre_state_batch, action_batch, post_state_batch, reward_batch, terminal_batch):
        '''
        需要注意,pre state和post State都需要作为输入,得到output pred y
        q targets是从post state出来的,q acted是从pre state出来的!
        有两个模型,并且权重需要相同!
        解决方法:将预测的那个网络作为placeholder!
        '''

        if not self.tf_weights_inited:
            raise RuntimeError("Please run init tf weights or load tf weights first!")
        # 计算q targets
        q_targets = self.sess.run(self.q_targets,
                                  feed_dict={

                                      self.input_state: post_state_batch,
                                      self.input_is_done: terminal_batch,
                                      self.input_rewards: reward_batch

                                  })
        # 然后进行训练,因为这里的训练操作还需要用到网络
        # print(self.q_targets_placeholder.shape,q_targets.shape)
        # print(self.input_state.shape,pre_state_batch.shape)
        # print(self.input_actions_1_hot.shape,action_batch.shape)
        # print("Q targets calculated!")
        self.sess.run(self.train_step,
                      feed_dict={
                          self.q_targets_placeholder: q_targets,
                          self.input_state: pre_state_batch,
                          self.input_actions_1_hot: action_batch
                      })
        self.n_trained_steps += 1


class ActorCriticModel_v1(keras.Model):
    Author = "BaoChuan Wang"
    AllowImport = True

    # 评价模型,输入是X,应当是state,输出是评价价值value,以及policy模型中的概率分布,是action的概率分布,对应每个action的概率
    def __init__(self, state_size, action_size):
        super(ActorCriticModel_v1, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.dense1 = layers.Dense(256, activation='relu')

        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(256, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values


class A3C_Global_And_Local_Models_v1(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
        流程:
        分为主网络和子网络
        主网络有一个,子网络每个worker有一个
        每次worker得到训练数据后,整理value,state和action给主网络
        主网络计算梯度进行更新,更新完成之后,子网络从主网络中拉取权重,使用assign操作赋值权重!
        '''

    def __init__(self,
                 action_space,
                 observation_space,
                 scope,
                 tf_sess,
                 global_model=None,
                 is_global_model=False,
                 optimize_a=tf.train.RMSPropOptimizer(learning_rate=0.001, name='RMSPropA'),
                 optimize_c=tf.train.RMSPropOptimizer(learning_rate=0.001, name='RMSPropC'),
                 save_dir="./ac3_ckpt/",
                 entropy_beta=0.001
                 ):
        assert len(action_space) == 1 and len(observation_space) == 1, "Only support vector input now!"
        # 目前只支持一维向量的输入
        action_size = action_space[0]
        observation_size = observation_space[0]

        if not is_global_model:
            assert global_model is not None, "Should feed a global model!"
        self.save_dir = save_dir
        self.sess = tf_sess
        self.action_size = action_size
        self.observation_size = observation_size
        if is_global_model:  # 有全局网络和局部网络之分
            with tf.variable_scope(scope):
                # 全局网络输入
                self.s = tf.placeholder(tf.float32, [None, observation_size], 'S')
                # 全局网络输出,只需要a模型和c模型的参数,因为全局网络实际上就是单纯的输入-a和c的可训练变量-输出
                # 因为不涉及到训练过程,所以相对于局部模型,少了很多操作
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, observation_size], 'S')
                # 历史action
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                # value
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                # policy的action概率输出,以及value输出
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                # 减去计算误差
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    # critic的loss是平方loss
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    # Q * log（
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.a_his, action_size, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    # 这里的td不再求导，当作是常数
                    exp_v = log_prob * tf.stop_gradient(td)
                    # encourage exploration
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):  # 把主网络的参数赋予各子网络的操作,全局网络专门用来存储a和c模型的params!
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_model.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_model.c_params)]
                with tf.name_scope('push'):  # 使用子网络的梯度对主网络参数进行更新的操作
                    # 需要注意,所有的优化都是在主网络进行,子网络仅仅提供input,然后主网络计算梯度!
                    self.update_a_op = optimize_a.apply_gradients(zip(self.a_grads, global_model.a_params))
                    self.update_c_op = optimize_c.apply_gradients(zip(self.c_grads, global_model.c_params))
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            # l_a = tf.layers.dense(self.s, 512, tf.nn.selu, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(self.s, 64, tf.nn.selu, kernel_initializer=w_init, name='la')
            l_a1 = tf.layers.dense(l_a, 128, tf.nn.selu, kernel_initializer=w_init, name='la1')
            l_a2 = tf.layers.dense(l_a1, 64, tf.nn.selu, kernel_initializer=w_init, name='la2')
            a_prob = tf.layers.dense(l_a2, self.action_size, tf.nn.softmax, kernel_initializer=w_init,
                                     name='ap')  # 得到每个动作的选择概率
        with tf.variable_scope('critic'):
            # l_c = tf.layers.dense(self.s, 512, tf.nn.selu, kernel_initializer=w_init, name='lc')
            l_c = tf.layers.dense(self.s, 64, tf.nn.selu, kernel_initializer=w_init, name='lc')
            l_c1 = tf.layers.dense(l_c, 128, tf.nn.selu, kernel_initializer=w_init, name='lc1')
            l_c2 = tf.layers.dense(l_c1, 64, tf.nn.selu, kernel_initializer=w_init, name='lc2')
            v = tf.layers.dense(l_c2, 1, kernel_initializer=w_init, name='v')  # 得到每个状态的价值函数
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        # softmax的输出是action概率,这里是随机根据概率选取,TODO:在演示或者运行时,需要使用最大概率的行为(贪心)
        prob_weights = self.sess.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class DDPG_Model_v2(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    DDPG model v1的另一种实现,应该是和v1版本没有太大效果上的差异,更改了部分API调用
    '''
    def __init__(self,
                 # action size
                 a_dim,
                 # state size
                 s_dim,
                 # action值的边界,用于缩放到-1,1
                 a_bound,
                 memory_capacity=10000,
                 lr_a=0.001,
                 lr_c=0.002,
                 gamma=0.9,
                 TAU=0.01,
                 batch_size=32
                 ):
        self.batch_size, self.lr_a, self.lr_c, self.gamma, self.TAU, self.memory_capacity = \
            batch_size, \
            lr_a, \
            lr_c, \
            gamma, \
            TAU, \
            memory_capacity

        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            # 从state到action的Dense
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # 从下一个状态得到action
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            # 从当前状态和下一个状态得到q值
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def reset(self):
        pass

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


class DDPG_Model_v2_With_Reward_PreCorr(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = True
    '''
    与普通DDPG_Model的差异 
    普通DDPG_Model:在训练集中随机选取batchsize个数据,然后每个数据在模型里面求q=gamma*q_+reward
    这里:直接先根据从开始到done的reward所有，从最后一个开始往前使用gamma回溯，回溯完的值直接作为target q
    注意,DDPG可能需要非连续采样保证不过拟合，上述方式只是获得state到（reward）target q的数据集，因此可以打乱和采样
    
    差异主要在learn函数中
    '''

    def __init__(self,
                 # action size
                 a_dim,
                 # state size
                 s_dim,
                 # action值的边界,用于缩放到-1,1
                 a_bound,
                 memory_capacity=100000,
                 lr_a=0.001,
                 lr_c=0.002,
                 gamma=0.99,
                 TAU=0.01,
                 batch_size=32
                 ):

        self.batch_size, self.lr_a, self.lr_c, self.gamma, self.TAU, self.memory_capacity = \
            batch_size, \
            lr_a, \
            lr_c, \
            gamma, \
            TAU, \
            memory_capacity

        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            # 从state到action的Dense
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # 从下一个状态得到action
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            # 从当前状态和下一个状态,以及action,得到q值
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # 直接使用reward，不再预测q_
        q_target = self.R  # + self.gamma * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def reset(self):
        self.pointer = 0

    def learn(self):
        # "注意，这里的learn只适用于terminal的情况！

        # soft target replacement
        self.sess.run(self.soft_replace)

        # 选取所有有效的数据
        # indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        # bt = self.memory[indices, :]
        bt = self.memory[:self.pointer, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # 然后预先求q值
        br = br.reshape(-1)
        r_target_array = []
        r_target = 0
        # 从后往前加，乘以gamma
        for r in br[::-1]:
            r_target = r + self.gamma * r_target
            r_target_array.append(r_target)
        r_target_array.reverse()
        print(r_target_array)
        br = np.array(r_target_array).reshape(-1, 1)

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    @staticmethod
    def test_pre_calc_q_target():
        br = np.array([0.1, 0.1, 0.2, 0.4, 1])

        br = br.reshape(-1)
        r_target_array = []
        r_target = 0

        for r in br[::-1]:
            r_target = r + 0.9 * r_target
            r_target_array.append(r_target)
        r_target_array.reverse()
        print(r_target_array)


class DDPG_Model_v1(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = False

    def __init__(self,
                 # action维度,int
                 a_dim,
                 # state维度
                 s_dim,
                 # 对action求得的值进行缩放
                 action_scale=1.0,
                 memory_capacity=10000,
                 batch_size=32,
                 # learning rate
                 actor_net_lr=0.001,
                 critic_net_lr=0.002,
                 # reward向前传播衰减
                 gamma=0.9,
                 TAU=0.01):
        raise NotImplementedError("Update from DDPG_Global_And_Local_Models_v1")
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.actor_net_lr = actor_net_lr
        self.critic_net_lr = critic_net_lr
        self.gamma = gamma
        self.TAU = TAU

        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, action_scale,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S, )
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)  # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]  # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.actor_net_lr).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + self.gamma * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(self.critic_net_lr).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        # 直接得到回归的action值
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


class DDPG_Global_And_Local_Models_v1(IModel, TFSaveLoadComponent):
    Author = "BaoChuan Wang"
    AllowImport = True

    '''    
    debug:
    variable scope不能使用,否则报错
    此时,每个NN层的name就会冲突
    因此每层都需要有独特的名称
    但是需要注意,在local模型中会创建两次模型,后一次的name必须要和前一次的相同,因为要reuse
    所以,一个local模型的所有层的名称,最好都加上输入参数scope(相当于修复之前with tf variable scope不能使用的问题)
    
    '''

    @staticmethod
    def test_learn_with_different_importance():
        # 测试不同local模型使用不同的lr对global优化
        a_dim = 2
        s_dim = 2
        sess = tf.Session()
        global_ = DDPG_Global_And_Local_Models_v1(
            scope="global",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=True

        )
        no_update = DDPG_Global_And_Local_Models_v1(
            scope="local1",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=False,
            global_model=global_,
            importance=0.0
        )
        small_update = DDPG_Global_And_Local_Models_v1(
            scope="local2",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=False,
            global_model=global_,
            importance=0.01
        )
        big_update = DDPG_Global_And_Local_Models_v1(
            scope="local3",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=False,
            global_model=global_,
            importance=1000000000.0
        )
        print("Initial global weights")
        print(sess.run(global_.a_params[0][:2, :2]))
        # 拉取权重
        no_update.pull_global()
        small_update.pull_global()
        big_update.pull_global()

        feed_dict = {
            global_.S: np.ones((3, a_dim)),
            global_.a: np.ones((3, a_dim)),
            global_.R: np.ones((3, 1)),
            global_.S_: np.ones((3, a_dim))
        }
        print("Weights sum after small update")

        # print(sess.run(small_update.a_grads, feed_dict=feed_dict)[0][:2,:2])

        small_update.give_global_to_learn(feed_dict, debug=True)
        print(sess.run(global_.c_params[0])[:2, :2])

        print("Weights sum after no update")
        # print(sess.run(no_update.a_grads, feed_dict=feed_dict)[0][:2,:2])

        no_update.give_global_to_learn(feed_dict, debug=True)
        print(sess.run(global_.c_params[0])[:2, :2])

        print("Weights sum after big update")
        # print(sess.run(big_update.a_grads, feed_dict=feed_dict)[0][:2,:2])

        big_update.give_global_to_learn(feed_dict, debug=True)
        print(sess.run(global_.c_params[0])[:2, :2])

    @staticmethod
    def test_pull_push():
        raise NotImplementedError("修改了模型更新方式,这里的测试代码失效")
        a_dim = 2
        s_dim = 2
        sess = tf.Session()
        global_ = DDPG_Global_And_Local_Models_v1(
            scope="global",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=True
        )
        local1 = DDPG_Global_And_Local_Models_v1(
            scope="local1",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=False,
            global_model=global_
        )
        local2 = DDPG_Global_And_Local_Models_v1(
            scope="local2",
            tf_sess=sess,
            a_dim=a_dim,
            s_dim=s_dim,
            is_global_model=False,
            global_model=global_
        )
        print("Initial global weights")
        print(sess.run(global_.a_params[0][:2, :2]))
        print("Initial local1 weight, should not be same as global")
        print(sess.run(local1.a_params[0][:2, :2]))
        local1.pull_global()
        local2.pull_global()
        print("local1 weight after pulling, should be same as global")
        print(sess.run(local1.a_params[0][:2, :2]))
        feed_dict = {
            local1.S: np.random.random(size=(3, a_dim)) * 1000,
            local1.a: np.random.random(size=(3, a_dim)) * 1000,
            local1.R: np.random.random(size=(3, 1)) * 1000,
            local1.S_: np.random.random(size=(3, a_dim)) * 1000
        }

        print("a loss")
        print(sess.run(local1.a_loss, feed_dict=feed_dict))
        # TODO:为什么这里a没有梯度?注意在RL运行时打印梯度!

        print("a Grads")
        print(sess.run(local1.a_grads, feed_dict=feed_dict))
        # print("a params")
        # print(sess.run(local1.a_params, feed_dict=feed_dict))
        print("c Grads")
        print(sess.run(local1.c_grads, feed_dict=feed_dict))
        print("Start push to global from local1")
        local1.update_to_global(feed_dict=feed_dict)
        print("global changed to")
        print(sess.run(global_.a_params[0][:2, :2]))

    def __init__(self,
                 # action维度,int
                 a_dim,
                 # state维度
                 s_dim,
                 # string
                 scope,
                 # 传入sess以保证各个sess一致,否则权重不会同步
                 tf_sess,
                 # 梯度更新的权重,相当于修改学习率,越大表明模型越容易从这里学习
                 # 需要注意,某些优化器无视loss,固定传递一定大小的梯度,此时importance无效
                 importance=1.0,
                 global_model=None,
                 is_global_model=False,
                 save_dir="./ddpg_gal_ckpt/",
                 # 对action求得的值进行缩放
                 action_scale=1.0,
                 memory_capacity=10000,
                 batch_size=128,
                 # optimize_a=tf.train.AdamOptimizer(0.0001),
                 # optimize_c=tf.train.AdamOptimizer(0.0002),
                 optimize_a=tf.train.RMSPropOptimizer(0.001),
                 optimize_c=tf.train.RMSPropOptimizer(0.002),
                 # reward向前传播衰减
                 gamma=0.9,
                 # 是否预先在加入数据集前计算q值,还是通过计算图来计算q值,具体查看learn中的pre calculate g的算法
                 use_pre_calculated_g=False,
                 TAU=0.01):
        if not is_global_model:
            assert global_model is not None, "Should feed a global model!"
        else:
            assert global_model is None, "Global Model should be None if it is not a global model"
        self.global_model = global_model
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.scope = scope
        self.gamma = gamma
        self.TAU = TAU
        self.use_pre_calculated_g = use_pre_calculated_g
        if use_pre_calculated_g:
            print("将预先根据reward求取q值,注意每次done过后需要清空memory")
        self.save_dir = save_dir
        self.importance = importance
        if (self.importance > 1.001) or (self.importance < 0.99):
            print("NOTICE:如果希望importance生效,请注意使用支持的optimizer by %s" % self.__class__.__name__)
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf_sess
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, action_scale,

        if is_global_model:
            # with tf.variable_scope(scope):
            # global 模型,最终需要a params和c params
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')
            self.target_a = tf.placeholder(tf.float32, (None, a_dim), "target_action")
            # 从state中预测action
            self.a = self._build_a(self.S, )
            # 根据预测出的action和当前state求q值
            q = self._build_c(self.S, self.a, )
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + 'Actor')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + 'Critic')

            ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [ema.apply(self.a_params), ema.apply(self.c_params)]  # soft update operation
            # 这里的S_是下一个state
            a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

            self.error_mul = tf.placeholder(shape=(1,), dtype=tf.float32)

            self.a_loss = - tf.reduce_mean(q)  # maximize the q
            self.a_loss = self.error_mul * self.a_loss
            self.atrain = optimize_a.minimize(self.a_loss, var_list=self.a_params)

            with tf.control_dependencies(target_update):  # soft replacement happened at here
                # pre calculated 已经计算过了,不需要再使用gamma
                if self.use_pre_calculated_g:
                    q_target = self.R
                else:
                    q_target = self.R + self.gamma * q_
                self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
                self.td_error = self.td_error * self.error_mul
                self.ctrain = optimize_c.minimize(self.td_error, var_list=self.c_params)

        else:
            # with tf.variable_scope(scope):
            # local 模型,最终给出梯度
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            self.a = self._build_a(self.S, )

            # 因为修改了global local更新逻辑,local只提供数据和拉取权重,因此少了loss求取的操作

            q = self._build_c(self.S, self.a, )
            self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + 'Actor')
            self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + 'Critic')

            # 下面的代码是另一种权重更新的方法,暂时保留

            # ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)  # soft replacement
            # """
            # 滑动平均(exponential moving average)，或者叫做指数加权平均(exponentially weighted moving average)，
            # 可以用来估计变量的局部均值，使得变量的更新与一段时间内的历史取值有关。
            # """
            #
            # def ema_getter(getter, name, *args, **kwargs):
            #     # 替换custom getter,详见tf文档custom getter
            #     return ema.average(getter(name, *args, **kwargs))
            #
            # target_update = [ema.apply(self.a_params), ema.apply(self.c_params)]  # soft update operation
            # # 注意,这里新创建的层和之前创建的name相同,所以reuse
            # a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
            # q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)
            #
            # # 得到a_loss和c_loss
            # self.a_loss = - tf.reduce_mean(q)  # maximize the q
            # with tf.control_dependencies(target_update):  # soft replacement happened at here
            #     q_target = self.R + self.gamma * q_
            #     self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

            # # 最终local得到grads
            # with tf.name_scope(self.scope+"local_grad"):
            #     self.a_grads = tf.gradients(self.a_loss, self.a_params)
            #     self.c_grads = tf.gradients(self.td_error, self.c_params)
            #     self.new_a_grads = []
            #     self.new_c_grads = []
            #
            #     for i in self.a_grads:
            #        self.new_a_grads.append(i*importance)
            #     for i in self.c_grads:
            #        self.new_c_grads.append(i*importance)
            #
            #     self.a_grads = self.new_a_grads
            #     self.c_grads = self.new_c_grads

            # pull操作,push操作在后面函数中
            with tf.name_scope(self.scope + 'sync'):
                with tf.name_scope(self.scope + "pull"):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_model.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_model.c_params)]
                # with tf.name_scope(self.scope+'push'):
                # self.update_a_op = optimize_a.apply_gradients(zip(self.a_grads, global_model.a_params))
                # self.update_c_op = optimize_c.apply_gradients(zip(self.c_grads, global_model.c_params))

        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

    def clear_memory(self):
        # 重新开始记录,但是不清空Memory
        self.pointer = 0

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    # def update_to_global(self, feed_dict):
    #     self.sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)

    def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def give_global_to_learn(self, feed_dict_replace=None,
                             # 误差修正系数,通常用于体现importance,变相调整学习率!
                             error_multiplier_replce=None,
                             debug=False):
        if error_multiplier_replce is None:
            error_mul = self.importance
        else:
            error_mul = error_multiplier_replce

        if feed_dict_replace is None:

            if self.use_pre_calculated_g:
                bt = self.memory[:self.pointer, :]
            else:
                # 下面有三种数据集选择方式

                # 原来随机取Memory里面的batchsize个,但是很多是全0的数据集
                # indices = np.random.choice(self.memory_capacity, size=self.batch_size)
                # bt = self.memory[indices,:]

                # 只允许拿出指针之前的dataset,因为只有这一部分是有效的,其他都是全0!
                indices = np.random.choice(min(self.pointer, self.memory_capacity - 1), size=self.batch_size)
                bt = self.memory[indices, :]

                # 拿出pointer前面全部的数据来训练!
                # bt = self.memory[:self.pointer, :]

            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            if self.use_pre_calculated_g:  # TODO:将求得q值的数据集加入buffer,而不是全部用来训练!
                br = br.reshape(-1)
                r_target_array = []
                r_target = 0
                # 从后往前加，乘以gamma
                for r in br[::-1]:
                    r_target = r + self.gamma * r_target
                    r_target_array.append(r_target)
                r_target_array.reverse()
                print(r_target_array)
                br = np.array(r_target_array).reshape(-1, 1)

            # print("Dataset:",bs[0,:],ba[0,:],br[0,:])
            feed_dict = {self.global_model.S: bs, self.global_model.a: ba, self.global_model.R: br,
                         self.global_model.S_: bs_}
        else:
            feed_dict = feed_dict_replace

        feed_dict[self.global_model.error_mul] = np.array([error_mul])
        # 开启debug
        debug = 0
        if debug:
            print("a loss and c loss")
            print(self.sess.run([self.global_model.a_loss, self.global_model.td_error], feed_dict))
        # 最新修改:直接求loss给全局模型
        self.sess.run([self.global_model.atrain, self.global_model.ctrain], feed_dict=feed_dict)

        # self.update_to_global(feed_dict)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(self.scope + 'Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 256, activation=tf.nn.relu, name=self.scope + 'l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim,
                                # 回归这里尝试不加激活函数
                                activation=tf.nn.tanh,
                                name=self.scope + 'a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name=self.scope + 'scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(self.scope + 'Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256
            w1_s = tf.get_variable(self.scope + 'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(self.scope + 'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable(self.scope + 'b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


if __name__ == '__main__':
    # DDPG_Global_And_Local_Models_v1.test_pull_push()
    DDPG_Global_And_Local_Models_v1.test_learn_with_different_importance()
    # DDPG_Model_v2_With_Reward_PreCorr.test_pre_calc_q_target()
