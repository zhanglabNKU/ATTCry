# -*- coding: UTF-8 -*-
# @Time    : 2020/8/25 下午3:27
# @Author  : Jin Chen
import keras
import numpy as np
from sklearn import metrics
from keras.optimizers import Adam
from keras.layers import *
from keras.utils import CustomObjectScope
from keras import backend as K
import tensorflow as tf

np.random.seed(1337)
epochs = 1
batch_size = 64


def TP(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred).ravel()[0]


def TN(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred).ravel()[3]


def FP(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred).ravel()[1]


def FN(y_true, y_pred): return metrics.confusion_matrix(y_true, y_pred).ravel()[2]


def MCC(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def Specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fn = FN(y_true, y_pred)
    return tp / (tp + fn)


def NPV(y_true, y_pred):
    tn = TN(y_true, y_pred)
    fn = FN(y_true, y_pred)
    return tn / (tn + fn)


def AUC(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def Precision(y_true, y_pred):
    precision = tf.metrics.precision(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return precision


def Recall(y_true, y_pred):
    recall = tf.metrics.recall(y_true, y_pred)[0]
    K.get_session().run(tf.local_variables_initializer())
    return recall


class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
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


# with CustomObjectScope({'Self_Attention': Self_Attention}):
model = keras.models.load_model('./model/bestmodel.h5', custom_objects={'Self_Attention': Self_Attention})

model.summary()
type = ['TR_Final_set', 'SP_Final_set', 'Balanced_Test_set']
num = 2
test_data = np.load('./Data/' + type[num] + '/x_test.npy')
test_label = np.load('./Data/' + type[num] + '/y_test.npy')

x_test = test_data
y_test = test_label

y_pred_net = model.predict(x_test)
np.save('./result/Main_y_score_mymodel1.npy', y_pred_net.transpose()[0])
result = []

for i in range(len(y_pred_net)):
    if y_pred_net[i][0] >= 0.5:
        result.append(1)
    else:
        result.append(0)
result = np.array(result)

print(metrics.classification_report(y_test, result))
print(metrics.confusion_matrix(y_test, result))
print('AUROC:', metrics.roc_auc_score(y_test, y_pred_net))
print('AUPR:', metrics.average_precision_score(y_test, y_pred_net))
print('MCC:', metrics.matthews_corrcoef(y_test, result))
print('Accuracy:', metrics.accuracy_score(y_test, result))
print('Specificity:', Specificity(y_test, result))
print('recall:', metrics.recall_score(y_test, result))
print('NPV:', NPV(y_test, result))
print('Precision:', metrics.precision_score(y_test, result))
print('F1:', metrics.f1_score(y_test, result))
