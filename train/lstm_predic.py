#-*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from matplotlib import style
import pandas as pd

'''读入原始数据并转为list'''
path = 'D:\\学习资料\\代码\\PCAP_analysis\\data\\'

data = pd.read_csv(path+'dataset_1.csv')

data = data.iloc[:,1].tolist()

'''自定义数据尺度缩放函数'''
def data_processing(raw_data,scale=True):
    if scale == True:
        return (raw_data-np.mean(raw_data))/np.std(raw_data)#标准化
    else:
        return (raw_data-np.min(raw_data))/(np.max(raw_data)-np.min(raw_data))#极差规格化

'''观察数据'''

'''设置绘图风格'''
style.use('ggplot')

plt.plot(data)

'''设置隐层神经元个数'''
HIDDEN_SIZE = 40
'''设置隐层层数'''
NUM_LAYERS = 1
'''设置一个时间步中折叠的递归步数'''
TIMESTEPS = 20
'''设置训练轮数'''
TRAINING_STEPS = 10000
'''设置训练批尺寸'''
BATCH_SIZE = 20




'''样本数据生成函数'''
def generate_data(seq):
    X = []#初始化输入序列X
    Y= []#初始化输出序列Y
    '''生成连贯的时间序列类型样本集，每一个X内的一行对应指定步长的输入序列，Y内的每一行对应比X滞后一期的目标数值'''
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i + TIMESTEPS]])#从输入序列第一期出发，等步长连续不间断采样
        Y.append([seq[i + TIMESTEPS]])#对应每个X序列的滞后一期序列值
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


'''定义LSTM cell组件，该组件将在训练过程中被不断更新参数'''
def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)#
    return lstm_cell

'''定义LSTM模型'''
def lstm_model(X, y):
    '''以前面定义的LSTM cell为基础定义多层堆叠的LSTM，我们这里只有1层'''
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])

    '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    '''根据预定义的每层神经元个数来生成隐层每个单元'''
    output = tf.reshape(output, [-1, HIDDEN_SIZE])

    '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    '''统一预测值与真实值的形状'''
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    '''定义损失函数，这里为正常的均方误差'''
    loss = tf.losses.mean_squared_error(predictions, labels)

    '''定义优化器各参数'''
    train_op = tf.contrib.layers.optimize_loss(loss,
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adagrad',
                                               learning_rate=0.6)
    '''返回预测值、损失函数及优化器'''
    return predictions, loss, train_op

'''载入tf中仿sklearn训练方式的模块'''
learn = tf.contrib.learn

'''初始化我们的LSTM模型，并保存到工作目录下以方便进行增量学习'''
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir='D:\\学习资料\\代码\\PCAP_analysis\\models\\model_2'))

'''对原数据进行尺度缩放'''
data = data_processing(data)

'''将所有样本来作为训练样本'''
train_X, train_y = generate_data(data[:])

'''将所有样本作为测试样本'''
test_X, test_y = generate_data(data[:])

'''以仿sklearn的形式训练模型，这里指定了训练批尺寸和训练轮数'''
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

'''利用已训练好的LSTM模型，来生成对应测试集的所有预测值'''
predicted = np.array([pred for pred in regressor.predict(test_X)])

'''绘制反标准化之前的真实值与预测值对比图'''
plt.figure()
plt.plot(predicted, label=u'预测值')
plt.plot(test_y, label=u'真实值')
plt.title(u'反标准化之前')
plt.legend()
plt.show()


'''自定义反标准化函数'''
def scale_inv(raw_data,scale=True):
    '''读入原始数据并转为list'''
    path = 'D:\\学习资料\\代码\\PCAP_analysis\\data\\'

    data = pd.read_csv(path + 'dataset_1.csv')

    data = data.iloc[:, 1].tolist()

    if scale == True:
        return raw_data*np.std(data)+np.mean(data)
    else:
        return raw_data*(np.max(data)-np.min(data))+np.min(data)


'''绘制反标准化之前的真实值与预测值对比图'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.plot(scale_inv(predicted), label='预测值')
plt.plot(scale_inv(test_y), label='真实值')
plt.title('反标准化之后')
plt.legend()
plt.show()