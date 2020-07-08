#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio

from odm import libodm, parameter, problem, read_libsvm_format, toPyFormat

# read data from libsvm format
y_train, X_train = read_libsvm_format('../data/svmguide1_train', True)
y_test, X_test = read_libsvm_format('../data/svmguide1_test', True)

prob_train = problem(y_train, X_train, bias=1)
prob_test = problem(y_test, X_test, bias=1)

# -s 0 dual coordinate descent
param = parameter('-s 0 -k 0 -l 64 -m 0.4 -t 0.9 -e 0.001')
model = libodm.train(prob_train, param)
prediction = libodm.predict(prob_test, model)

# -s 1 trust region Newton method
param = parameter('-s 1 -k 0 -l 128 -m 0.1 -t 0.9 -e 0.001')
model = libodm.train(prob_train, param)
prediction = libodm.predict(prob_test, model)

param = parameter('-s 2 -k 0 -l 64 -m 0.2 -t 0.8 -e 0.001')  # -s 2 svrg
model = libodm.train(prob_train, param)
prediction = libodm.predict(prob_test, model)

prob_train = problem(y_train, X_train, bias=0)
prob_test = problem(y_test, X_test, bias=0)

# -k 2 rbf kernel
param = parameter('-s 0 -k 2 -l 4096 -m 0.3 -t 0.1 -g 16 -e 0.001')
model = libodm.train(prob_train, param)
prediction = libodm.predict(prob_test, model)
# p = toPyFormat(prediction)
# print(p.pre_acc)

# read data from octave sparse matrix
# data = scio.loadmat('../data/svmguide1_sparse.mat')
# y_train, X_train = np.squeeze(data['y_train']), data['X_train']
# y_test, X_test = np.squeeze(data['y_test']), data['X_test']
# prob_train = problem(y_train, X_train, bias=1)
# prob_test = problem(y_test, X_test, bias=1)


# read data from octave dense matrix
# data = scio.loadmat('../data/svmguide1.mat')
# y_train, X_train = np.squeeze(data['y_train']), data['X_train']
# y_test, X_test = np.squeeze(data['y_test']), data['X_test']
# prob_train = problem(y_train, X_train, bias=1)
# prob_test = problem(y_test, X_test, bias=1)
