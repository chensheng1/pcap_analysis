import tensorflow as tf


tf.reset_default_graph()  # 清除默认图形堆栈并重置全局默认图形.

class M_LSTM():
    def main(self,texts_train1,text_test1,target_train,target_test):
        # 设置用到的参数
        lr0 = 0.001
        global_step = tf.Variable(0)
        lr_decay = 0.99
        lr_step = 500
        #输入数据
        x=188
        # 在训练和测试的时候 想使用不同的batch_size 所以采用占位符的方式
        batch_size = tf.placeholder(tf.int32, [])
        # 每个隐含层的节点数
        hidden_size = 30
        # LSTM的层数
        layer_num = 9
        # 最后输出的分类类别数量，如果是回归预测的呼声应该是1
        class_num = 9
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
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([unit_lstm() for i in range(3)], state_is_tuple=True)

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

        # 开始训练

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _batch_size = 10
            start = (i * _batch_size) % texts_train1.shape[0]  #
            end = start + _batch_size
            batch_len = texts_train1[start:end].shape[0]
            batch_x = texts_train1[start:end].reshape((batch_len, 1, x))
            batch_y = target_train[start:end].reshape((batch_len, layer_num))
            sess.run([train_op], feed_dict={_X: batch_x, y: batch_y,keep_prob: 1.0,batch_size: _batch_size})
            if (i + 1) % 200 == 0:
                loss,train_accuracy = sess.run([cross_entropy,accuracy], feed_dict={
                    _X: batch_x, y: batch_y, keep_prob: 1.0, batch_size: _batch_size
                })
                print("step %d,loss %s, training accuracy %g" % ((i + 1), loss,train_accuracy))
        saver.save(sess, "D:\\学习资料\\代码\\PCAP_analysis\\model\\model_bin.ckpt")
        len1=len(text_test1)
        #print("test accuracy %g" % sess.run(accuracy, feed_dict={_X: text_test1[:].reshape((len1,1,x)), y:target_test.reshape((len1, layer_num)), keep_prob: 1.0,
        #                                                         batch_size: len1}))
        list = sess.run(y_pre, feed_dict={_X: text_test1[:].reshape((len1, 1, x)),
                                          y: target_test.reshape((len1, layer_num)), keep_prob: 1.0,
                                          batch_size: len1})
        dict={'Analysis':0,'Backdoor':1,'Dos':2,'Exploits':3,'Fuzzers':4,'Generic':5,'Reconnaissance':6,'Shellcode':7,'Worms':8}
        result = []
        for i in list.tolist():
            max_index=i.index(max(i))
            for key,values in dict.items():
                if values==max_index:
                    result.append(key+"概率:" + str(i[max_index]))
        print(result)