# Type: Private Author: Baochuan Wang


# __________ tests ______________


def dense_q_eval_model_fake_train():
    model = DenseModel(
        input_space_size=(4,),
        output_space_size=(5,), ckpt_directory="ckpt/")
    batch_size = 10
    pre_state_batch = np.random.random((batch_size, 4))
    post_state_batch = np.random.random((batch_size, 4))
    reward_batch = np.random.random((batch_size, 1))
    # 随机选取01
    terminal_batch = np.random.choice(1, (batch_size, 1)).astype("float32")
    actions = np.random.choice([4], (batch_size, 1))
    action_one_hot = np.zeros((batch_size, 5))
    action_one_hot[:, actions] = 1
    actions_batch = action_one_hot

    model.init_tf_weights()
    try:
        model.load_tf_weights()
    except ValueError:
        pass

    for _ in range(10):
        print(_)
        model.train(pre_state_batch=pre_state_batch,
                    post_state_batch=post_state_batch,
                    terminal_batch=terminal_batch,
                    reward_batch=reward_batch,
                    action_batch=actions_batch)
    model.save_tf_weights()


def show_model():
    model = ConvFeatureConcatModel()
    model.build()
    model.show_graph_shape()

