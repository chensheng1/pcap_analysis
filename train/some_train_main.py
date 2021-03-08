
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import multi_lstm_model

def load_data():
    # Default values.
    train_set = 'D:\\学习资料\\代码\\PCAP_analysis\\data\\training.csv'
    test_set = 'D:\\学习资料\\代码\\PCAP_analysis\\data\\testing.csv'
    data1 = pd.read_csv(train_set, index_col='id')
    train = shuffle(data1[data1['label']==1])
    data2 = pd.read_csv(test_set, index_col='id')
    test1 = shuffle(data2[data2['label'] == 1])
    test=test1[test1['label']==1]
    temp_train1 = pd.get_dummies(train['attack_cat'],drop_first=False)
    print(temp_train1)
    temp_train=temp_train1.values

    temp_test1 = pd.get_dummies(test['attack_cat'], drop_first=False)
    temp_test=temp_test1.values


    # Creates new dummy columns from each unique string in a particular feature 创建新的虚拟列
    unsw = pd.concat([train, test])
    unsw = pd.get_dummies(data=unsw, columns=['proto', 'service', 'state'])
    print(unsw)
    # Normalising all numerical features:
    unsw.drop(['label', 'attack_cat'], axis=1, inplace=True)
    unsw_value = unsw.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    unsw_value = scaler.fit_transform(unsw_value)
    train_set = unsw_value[:len(train), :]
    test_set = unsw_value[len(train):, :]

    return train_set, np.array(temp_train), test_set, np.array(temp_test)



def divicde(data1,label):
    train_indices = np.random.choice(data1.shape[0], round(0.8 * data1.shape[0]),
                                     replace=False)
    test_indices = np.array(list(set(range(data1.shape[0])) - set(train_indices)))
    texts_train1 = data1[train_indices]
    texts_test1 = data1[test_indices]
    #texts_train2 = np.array([y for iy, y in enumerate(data2) if iy in train_indices])
    #texts_test2 = np.array([y for iy, y in enumerate(data2) if iy in test_indices])
    target_train = np.array([y for iy, y in enumerate(label) if iy in train_indices])
    target_test = np.array([y for iy, y in enumerate(label) if iy in test_indices])
    return texts_train1,texts_test1,target_train,target_test


if __name__ == '__main__':
    train_set, temp_train, test_set, temp_test = load_data()
    print(len(train_set[0]))
    print(temp_train)
    multi_lstm_model.M_LSTM().main(train_set, test_set,temp_train, temp_test)