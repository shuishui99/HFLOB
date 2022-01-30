import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report
import logging

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    all_label = []

    for i in range(lob.shape[1]):
        one_label = lob[:, i] - 1
        one_label = keras.utils.to_categorical(one_label, 3)
        one_label = one_label.reshape(len(one_label), 1, 3)
        all_label.append(one_label)

    return np.hstack(all_label)


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


def create_dataset(x_train, y_train, batch_size, method, shuffle=False):
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(len(x_train))
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(lambda d, l: (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

    if method == 'train':
        return train_ds.repeat()

    if method == 'val':
        return train_ds

    if method == 'test':
        return train_ds

    if method == 'prediction':
        train_ds = tf.data.Dataset.from_tensor_slices((x_train))
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda d: (tf.cast(d, tf.float32)))
        return train_ds


def evaluation_metrics(real_y, pred_y):
    real_y = real_y[:len(pred_y)]
    logging.info('-------------------------------')

    for i in range(real_y.shape[1]):
        print(f'Prediction horizon = {i}')
        print(f'accuracy_score = {accuracy_score(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1))}')
        print(f'classification_report = {classification_report(np.argmax(real_y[:, i], axis=1), np.argmax(pred_y[:, i], axis=1), digits=4)}')
        print('-------------------------------')


def prepare_decoder_input(data, teacher_forcing):
    if teacher_forcing:
        first_decoder_input = keras.utils.to_categorical(np.zeros(len(data)), 3)
        first_decoder_input = first_decoder_input.reshape(len(first_decoder_input), 1, 3)
        decoder_input_data = np.hstack((data[:, :-1, :], first_decoder_input))

    if not teacher_forcing:
        decoder_input_data = np.zeros((len(data), 1, 3))
        decoder_input_data[:, 0, 0] = 1.

    return decoder_input_data


if __name__ == '__main__':
    T = 50
    epochs = 50
    batch_size = 32
    n_hidden = 64
    dec_train = np.loadtxt('/home/yuan/jam/pycharm/data/Train_Dst_NoAuction_DecPre_CF_7.txt')
    print("done!")
    # dec_test1 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_test2 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_8.txt')
    # dec_test3 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_9.txt')
    # dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    # extract limit order book data from the FI-2010 dataset
    train_lob = prepare_x(dec_train)
    # test_lob = prepare_x(dec_test)

    # extract label from the FI-2010 dataset
    train_label = get_label(dec_train)
    # test_label = get_label(dec_test)

    # prepare training data. We feed past 100 observations into our algorithms.
    train_encoder_input, train_decoder_target = data_classification(train_lob, train_label, T)
    train_decoder_input = prepare_decoder_input(train_encoder_input, teacher_forcing=False)
    print(f'train_encoder_input.shape = {train_encoder_input.shape},'
          f'train_decoder_target.shape = {train_decoder_target.shape}')

    # test_encoder_input, test_decoder_target = data_classification(test_lob, test_label, T)
    # test_decoder_input = prepare_decoder_input(test_encoder_input, teacher_forcing=False)