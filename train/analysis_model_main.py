import tensorflow as tf
import processing

class LSTM_model:

    def main(self,texts_train1,text_test1,target_train,target_test):
        # 设置用到的参数
        lr0 = 0.001
        global_step = tf.Variable(0)
        lr_decay = 0.99
        lr_step = 500
        #输入数据
        x=196
        # 在训练和测试的时候 想使用不同的batch_size 所以采用占位符的方式
        batch_size = tf.placeholder(tf.int32, [])
        # 每个隐含层的节点数
        hidden_size = 20
        # LSTM的层数
        layer_num = 2
        # 最后输出的分类类别数量，如果是回归预测的呼声应该是1
        class_num = 2
        _X = tf.placeholder(tf.float32, [None,1, x])
        y = tf.placeholder(tf.float32, [None, class_num])
        keep_prob = tf.placeholder(tf.float32)
        lr = tf.train.exponential_decay(
            lr0,
            global_step,
            decay_steps=lr_step,
            decay_rate=lr_decay,
            staircase=True)
        # 定义一个LSTM结构， 把784个点的字符信息还原成28*28的图片
        X = tf.reshape(_X, [-1, 1, x])

        def unit_lstm():
            # 定义一层LSTM_CELL hiddensize 会自动匹配输入的X的维度
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
            # 添加dropout layer， 一般只设置output_keep_prob
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            return lstm_cell

        # 调用MultiRNNCell来实现多层 LSTM
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(2)], state_is_tuple=True)

        # 使用全零来初始化state
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state,
                                           time_major=False)
        h_state = outputs[:, -1, :]

        # 设置loss function 和优化器
        W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
        bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
        y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
        # 损失和评估函数
        cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
        train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        saver = tf.train.Saver()
        # 开始训练
        sess = tf.Session()
        saver.restore(sess, "D:\\学习资料\\代码\\PCAP_analysis\\model\\model_mu.ckpt")
        len1=len(text_test1)
        print("test accuracy %g" % sess.run(accuracy, feed_dict={_X: text_test1[:].reshape((len1,1,x)), y:target_test.reshape((len1, layer_num)), keep_prob: 1.0,
                                                                 batch_size: len1}))
        list=sess.run(y_pre, feed_dict={_X: text_test1[:].reshape((len1, 1, x)),
                                                           y: target_test.reshape((len1, layer_num)), keep_prob: 1.0,
                                                         batch_size: len1})
        result=[]
        for i in list:
            if(i[0]>i[1]):
                result.append("正常概率:"+str(i[0]))
            else:
                result.append("异常概率:"+str(i[1]))
        print(result)

if __name__ == '__main__':
    train_set, temp_train, test_set, temp_test=processing.load_data()
    LSTM_model().main(train_set, test_set,temp_train, temp_test)