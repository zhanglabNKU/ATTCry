# -*- coding: UTF-8 -*-
# @Time    : 19-6-8 下午9:21
# @Author  : Jin Chen
import numpy as np
import pandas as pd
from collections import Counter


def str_supplement(num):
    str = ''
    for i in range(num):
        str += 'Z'
    return str


def get_data(file):
    fi = open(file, 'r')
    max_seq = 800
    seqs = []
    seq = ""
    labels = []

    for line in fi:
        if (line[0] == ">"):
            seqs.append(seq)
            seq = ""
        else:
            seq = seq + line.split('\n')[0]

    seqs = seqs[1:]

    print(len(seqs))

    dic = {'G': 0, 'A': 1, 'V': 2, 'L': 3, 'I': 4, 'F': 5, 'W': 6, 'Y': 7, 'D': 8, 'N': 9, 'E': 10, 'K': 11, 'Q': 12,
           'M': 13, 'S': 14, 'T': 15, 'C': 16, 'P': 17, 'H': 18, 'R': 19, 'B': 20, 'X': 20, 'J': 20, 'O': 20, 'U': 20,
           'Z': 20, '.': 20}
    seq_new = []
    for seq in seqs:
        if (len(seq) < max_seq):
            seq_new.append((seq + str_supplement(max_seq - len(seq))))
            # print(seq)
        else:
            seq_new.append(seq[:max_seq])

    a = []
    b = []
    for seq in seq_new:
        for amino in seq:
            a.append(dic.get(amino))
        b.append(a)
        a = []
    data = np.array(b, dtype=int)

    return data


def get_label(file):
    labels = []
    label = pd.read_csv(file, header=None).values
    # label = pd.read_csv(file).values
    for i in label:
        labels.append(int(i))
    labels = np.array(labels, dtype=int)
    return labels


if __name__ == '__main__':
    # type=['Balanced_Test_set','SP_Final_set','TR_Final_set']
    # num=1
    # data_file = './Data/'+type[num]+'/test.fasta'
    # label_file= './Data/'+type[num]+'/y_test.csv'
    # x_test=get_data(data_file)
    # y_test=get_label(label_file)
    # print(len(x_test),len(y_test))
    # print(Counter(y_test))
    # np.save('./Data/'+type+'/x_test', x_test)
    # np.save('./Data/'+type+'/y_test', y_test)

    type = 'Train'
    data_file = './Data/' + type + '/FULL_Train.fasta'
    label_file = './Data/' + type + '/Train_True_Labels.csv'
    x_train = get_data(data_file)
    y_train = get_label(label_file)
    print(len(x_train), len(y_train))
    print(Counter(y_train))
    # np.save('./Data/'+type+'/x_train1', x_train)
    # np.save('./Data/'+type+'/y_train1', y_train)
