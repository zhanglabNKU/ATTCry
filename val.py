# -*- coding: UTF-8 -*-
# @Time    : 19-9-4 下午2:59
# @Author  : Jin Chen
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam, Adagrad
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle, class_weight

import pandas as pd
from sklearn import metrics

np.random.seed(1337)
num_classes = 1
data_augmentation = True
l2value = 0.001
epochs = 2
batch_size = 64

train_data = np.load('./Data/Train/x_train.npy')
train_label = np.load('./Data/Train/y_train.npy')
train_index = np.arange(0, len(train_data))
np.random.shuffle(train_index)

x_train = train_data[train_index]
y_train = train_label[train_index]

class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weight_dict = dict(enumerate(class_weight))
auc_scores = []


class Self_Attention(Layer):

    def __init__(self, output_dim=64, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(Self_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

val_scores = []

for fold, (tra, val) in enumerate(kfold.split(x_train, y_train)):
    print('Train:%s' % fold)
    ########################################################################################################
    ########################################################################################################
    main_input = Input(shape=x_train.shape[1:], name='main_input')
    x = Embedding(output_dim=50, input_dim=21, input_length=800)(main_input)
    # ########################################################################################################
    cnn1 = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    cnn2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    cnn3 = Conv1D(64, kernel_size=8, activation='relu', padding='same')(x)
    cnn4 = Conv1D(64, kernel_size=9, activation='relu', padding='same')(x)
    cnn5 = Conv1D(64, kernel_size=4, activation='relu', padding='same')(x)
    cnn6 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    cnn7 = Conv1D(64, kernel_size=6, activation='relu', padding='same')(x)
    cnn8 = Conv1D(64, kernel_size=7, activation='relu', padding='same')(x)
    concat1 = concatenate([cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8], axis=-1)
    pool1 = MaxPooling1D(pool_size=5)(concat1)
    drop1 = Dropout(0.3)(pool1)
    ########################################################################################################
    cnnb1 = Conv1D(64, kernel_size=11, activation='relu', padding='same')(drop1)
    cnnb2 = Conv1D(64, kernel_size=13, activation='relu', padding='same')(drop1)
    cnnb3 = Conv1D(64, kernel_size=15, activation='relu', padding='same')(drop1)
    concat2 = concatenate([cnnb1, cnnb2, cnnb3], axis=-1)
    pool2 = MaxPooling1D(pool_size=5)(concat2)
    drop2 = Dropout(0.3)(pool2)
    # ########################################################################################################
    O_seq1 = Self_Attention(32)(drop2)
    O_seq2 = Self_Attention(32)(drop2)
    O_seq3 = Self_Attention(32)(drop2)
    O_seq4 = Self_Attention(32)(drop2)
    O_seq5 = Self_Attention(32)(drop2)
    O_seq6 = Self_Attention(32)(drop2)
    con1 = concatenate([O_seq1, O_seq2, O_seq3, O_seq4, O_seq5, O_seq6], axis=-1)
    ########################################################################################################
    drop_a = Dropout(0.3)(con1)
    flat1 = Flatten()(drop_a)
    dense1 = Dense(1024, activation='relu', kernel_regularizer=l2(l2value))(flat1)
    drop3 = Dropout(0.3)(dense1)
    dense2 = Dense(128, activation='relu', kernel_regularizer=l2(l2value))(drop3)
    drop4 = Dropout(0.3)(dense2)
    dense3 = Dense(16, activation='relu', kernel_regularizer=l2(l2value))(drop4)
    drop = Dropout(0.3)(dense3)
    output = Dense(num_classes, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(drop)
    ########################################################################################################
    model = Model(input=main_input, output=output)
    ########################################################################################################
    model.summary()
    opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)
    adam = Adam(lr=0.001)
    adagrad = Adagrad(lr=0.01)
    model.compile(
        loss='binary_crossentropy',
        optimizer=adam,
        metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             batch_size=32,  # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)
    fit = model.fit(x_train[tra], y_train[tra], validation_data=(x_train[val], y_train[val]), nb_epoch=100,
                    batch_size=batch_size, shuffle=True, callbacks=[early_stopping, tbCallBack])
    print(early_stopping.stopped_epoch)
    acc = model.evaluate(x_train[val], y_train[val])[1]
    print(acc)
    val_scores.append(acc)
    model.save('./fold/model_' + str(fold) + '.h5')
print(np.mean(val_scores))
