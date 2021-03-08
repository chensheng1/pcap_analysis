import LSTM_model
import processing

if __name__ == '__main__':
    train_set, temp_train, test_set, temp_test=processing.load_data()
    print(train_set[0])
    print(temp_train)
    LSTM_model.LSTM_model().main(train_set, test_set,temp_train, temp_test)
