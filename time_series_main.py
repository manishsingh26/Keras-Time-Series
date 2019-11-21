import time
import numpy as np
import pandas as pd
from numpy import newaxis
from subprocess import check_output
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout


class DataAnalysis(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)

        yahoo = df[df['symbol'] == 'YHOO']
        yahoo_stock_prices = yahoo.close.values.astype('float32')
        yahoo_stock_prices = yahoo_stock_prices.reshape(1762, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        yahoo_stock_prices = scaler.fit_transform(yahoo_stock_prices)

        train_size = int(len(yahoo_stock_prices) * 0.80)
        test_size = len(yahoo_stock_prices) - train_size
        train, test = yahoo_stock_prices[0:train_size, :], yahoo_stock_prices[train_size:len(yahoo_stock_prices), :]

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return trainX, trainY, testX, testY


class NeuralConfiguration(object):

    def __init__(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def model_creation(self):

        print("Creating Sequential Model.")
        model = Sequential()

        model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
        model.add(Dropout(rate=0.2))

        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(rate=0.2))

        model.add(Dense(output_dim=1))
        model.add(Activation('linear'))

        print("Adding Optimizer in the Model.")
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])

        print("Fitting Data in th Model.")
        model.fit(self.trainX, self.trainY, batch_size=128, nb_epoch=10, validation_split=0.05)

        print("Working on Model Evaluation.")
        _, accuracy = model.evaluate(self.trainX, self.trainY)
        print('Accuracy: %.2f' % (accuracy * 100))

        print("Predicting Value for the Test Data-set.")
        predicted_data = model.predict(np.array(self.testX))
        return predicted_data


if __name__ == "__main__":

    file_path_ = r"C:\Users\m4singh\Documents\AnalysisNoteBook\DeepLearning\Timeseries\prices.csv"

    file_obj = DataAnalysis(file_path_)
    trainX_, trainY_, testX_, testY_ = file_obj.load_data()

    neural_obj = NeuralConfiguration(trainX_, trainY_, testX_, testY_)
    predicted_data_ = neural_obj.model_creation()

    print(predicted_data_[:4])
