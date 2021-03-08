import pandas as pd
from keras.models import load_model

def read_path(path,path1,conf):
    data = pd.read_csv(path,header=0,index_col=1).shape
    m3 = load_model(conf)
    predicted = m3.predict(data)
    print(predicted)
    real=pd.read_csv(path1,header=0,index_col=0)
    result=[]
    for i in range(0,len(predicted)):
        if(real[i][0]==0):
            result.append("正常："+str(predicted[i][0]))
        else:
            result.append("异常：" + str(predicted[i][0]))
    return result


if __name__ == '__main__':
    result=read_path("C:\\Users\\asus\\Desktop\\test_data.csv","C:\\Users\\asus\\Desktop\\test_labels.csv","C:\\Users\\asus\\Desktop\\cnn_model.hdf5")
    print(result)

