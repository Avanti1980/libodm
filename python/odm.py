#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from array import array
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, addressof, c_char_p,
                    c_double, c_int, c_uint64, cast, pointer, sizeof)
from ctypes.util import find_library
from os import path

try:
    import scipy
    from scipy import sparse
except:
    scipy = None
    sparse = None

# odm.py interface
__all__ = ['libodm', 'feature_node', 'gen_feature_nodearray', 'problem', 'parameter', 'model',
           'toPyFormat', 'CD', 'TR', 'SVRG', 'LINEAR', 'POLY', 'RBF', 'SIGMOID', 'print_null', 'read_libsvm_format']

try:
    dirname = path.dirname(path.abspath(__file__))
    if sys.platform == 'win32':
        libodm = CDLL(path.join(dirname, r'..\windows\libodm.dll'))
    else:
        libodm = CDLL(path.join(dirname, '../libodm.so'))
except:
    if find_library('odm'):  # For unix the prefix 'lib' is not considered.
        libodm = CDLL(find_library('odm'))
    elif find_library('libodm'):
        libodm = CDLL(find_library('libodm'))
    else:
        raise Exception('libodm library not found.')

# solver
CD = 0
TR = 1
SVRG = 2

# kernel
LINEAR = 4
POLY = 5
RBF = 6
SIGMOID = 7

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)


def print_null(s):
    return


def genFields(names, types):
    return list(zip(names, types))


def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes


class feature_node(Structure):
    _names = ["index", "value"]
    _types = [c_int, c_double]
    _fields_ = genFields(_names, _types)

    def __str__(self):
        return '%d:%g' % (self.index, self.value)


def gen_feature_nodearray(xi, feature_max=None):
    if feature_max:
        assert(isinstance(feature_max, int))

    xi_shift = 0  # ensure correct indices of xi

    if scipy and isinstance(xi, tuple) and len(xi) == 2:  # tuple(index, value)
        if isinstance(xi[0], (list, tuple)):
            index_range = scipy.array(xi[0])
        else:
            index_range = xi[0]
        if feature_max:
            index_range = index_range[scipy.where(index_range <= feature_max)]

    elif scipy and isinstance(xi, scipy.ndarray):  # ndarray (1-D)
        xi_shift = 1
        index_range = xi.nonzero()[0] + 1  # nonzero return starts at 0
        if feature_max:
            index_range = index_range[scipy.where(index_range <= feature_max)]

    elif isinstance(xi, (dict, list, tuple)):
        if isinstance(xi, dict):
            index_range = xi.keys()
        elif isinstance(xi, (list, tuple)):
            xi_shift = 1
            index_range = range(1, len(xi) + 1)

        # discard zero entries
        index_range = filter(lambda j: xi[j-xi_shift] != 0, index_range)

        # discard features whose index larger than feature_max
        if feature_max:
            index_range = filter(lambda j: j <= feature_max, index_range)

        # filter function returns an iterator, sorted function converts it to list
        index_range = sorted(index_range)
    else:
        raise TypeError(
            'xi should be a dictionary, list, tuple, 1-d numpy array, or tuple of (index, data)')

    # init feature_node struct array
    ret = (feature_node*(len(index_range)+1))()
    ret[-1].index = -1  # for bias term
    ret[-2].index = -1

    # tuple(index, value)
    if scipy and isinstance(xi, tuple) and len(xi) == 2:
        for idx, j in enumerate(index_range):
            ret[idx].index = j
            ret[idx].value = xi[1][idx]

    # list / tuple / dict / ndarray (1-D)
    else:
        for idx, j in enumerate(index_range):
            ret[idx].index = j
            ret[idx].value = xi[j - xi_shift]

    max_idx = 0
    if len(index_range) > 0:
        max_idx = index_range[-1]
    return ret, max_idx


try:
    from numba import jit
    jit_enabled = True
except:
    def jit(x): return x
    jit_enabled = False


@jit
def csr_to_problem_jit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
    for i in range(l):
        b1, e1 = x_rowptr[i], x_rowptr[i+1]
        b2, e2 = prob_rowptr[i], prob_rowptr[i+1]-2
        for j in range(b1, e1):
            prob_ind[j-b1+b2] = x_ind[j]+1
            prob_val[j-b1+b2] = x_val[j]


def csr_to_problem_nojit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
    for i in range(l):
        x_slice = slice(x_rowptr[i], x_rowptr[i+1])
        prob_slice = slice(prob_rowptr[i], prob_rowptr[i+1]-2)
        prob_ind[prob_slice] = x_ind[x_slice]+1
        prob_val[prob_slice] = x_val[x_slice]


def csr_to_problem(x, prob):
    x_space = prob.x_space = scipy.empty(
        (x.nnz+2*x.shape[0]), dtype=feature_node)
    prob.rowptr = x.indptr.copy()
    prob.rowptr[1:] += 2*scipy.arange(1, x.shape[0]+1)
    prob_ind = x_space["index"]
    prob_val = x_space["value"]
    prob_ind[:] = -1
    if jit_enabled:
        csr_to_problem_jit(x.shape[0], x.data, x.indices,
                           x.indptr, prob_val, prob_ind, prob.rowptr)
    else:
        csr_to_problem_nojit(
            x.shape[0], x.data, x.indices, x.indptr, prob_val, prob_ind, prob.rowptr)


class problem(Structure):
    _names = ["m", "d", "bias", "y", "x"]
    _types = [c_int, c_int, c_int, POINTER(
        c_double), POINTER(POINTER(feature_node))]
    _fields_ = genFields(_names, _types)

    def __init__(self, y, x, bias=0):

        if (not isinstance(y, (list, tuple))) and (not (scipy and isinstance(y, scipy.ndarray))):
            raise TypeError("type of y: {0} is not supported!".format(type(y)))

        if isinstance(x, (list, tuple)):
            if len(y) != len(x):
                raise ValueError("len(y) != len(x)")

        elif scipy != None and isinstance(x, (scipy.ndarray, sparse.spmatrix)):
            if len(y) != x.shape[0]:
                raise ValueError("len(y) != len(x)")
            if isinstance(x, scipy.ndarray):
                x = scipy.ascontiguousarray(x)
            if isinstance(x, sparse.spmatrix):
                x = x.tocsr()
                pass
        else:
            raise TypeError("type of x: {0} is not supported!".format(type(x)))

        self.m = m = len(y)  # instance number

        self.bias = bias

        max_idx = 0
        x_space = self.x_space = []

        if scipy != None and isinstance(x, sparse.csr_matrix):
            csr_to_problem(x, self)
            max_idx = x.shape[1]
        else:
            for i, xi in enumerate(x):
                tmp_xi, tmp_idx = gen_feature_nodearray(xi)
                x_space += [tmp_xi]
                max_idx = max(max_idx, tmp_idx)

        self.d = max_idx  # dimension

        self.y = (c_double * m)()
        if scipy != None and isinstance(y, scipy.ndarray):  # ndarray (1-D)
            scipy.ctypeslib.as_array(self.y, (self.m,))[:] = y
        else:
            for i, yi in enumerate(y):  # list / tuple
                self.y[i] = yi

        self.x = (POINTER(feature_node) * m)()
        if scipy != None and isinstance(x, sparse.csr_matrix):
            base = addressof(self.x_space.ctypes.data_as(
                POINTER(feature_node))[0])
            x_ptr = cast(self.x, POINTER(c_uint64))
            x_ptr = scipy.ctypeslib.as_array(x_ptr, (self.m,))
            x_ptr[:] = self.rowptr[:-1]*sizeof(feature_node)+base
        else:
            for i, xi in enumerate(self.x_space):
                self.x[i] = xi

        if self.bias == 1:
            self.d += 1
            node = feature_node(self.d, 1)
            if isinstance(self.x_space, list):
                for xi in self.x_space:
                    xi[-2] = node
            else:
                self.x_space["index"][self.rowptr[1:]-2] = node.index
                self.x_space["value"][self.rowptr[1:]-2] = node.value


class parameter(Structure):
    _names = ["solver", "kernel", "lambda_", "mu", "theta",
              "degree", "gamma", "coef0", "frequency", "eps"]
    _types = [c_int, c_int, c_double, c_double, c_double,
              c_int, c_double, c_double, c_int, c_double]
    _fields_ = genFields(_names, _types)

    def __init__(self, options=None):
        if options == None:
            options = ''
        self.parse_options(options)

    def __str__(self):
        s = ''
        attrs = parameter._names + list(self.__dict__.keys())
        values = map(lambda attr: getattr(self, attr), attrs)
        for attr, val in zip(attrs, values):
            s += (' %s: %s\n' % (attr, val))
        s = s.strip()

        return s

    def set_to_default_values(self):
        self.solver = CD
        self.kernel = LINEAR
        self.lambda_ = 1
        self.mu = 0.8
        self.theta = 0.2
        self.degree = 2
        self.gamma = 0.1
        self.coef0 = 0.5
        self.frequency = 5
        self.eps = 0.01
        self.print_func = cast(None, PRINT_STRING_FUN)

    def parse_options(self, options):
        if isinstance(options, list):
            argv = options
        elif isinstance(options, str):
            argv = options.split()
        else:
            raise TypeError("arg 1 should be a list or a str.")
        self.set_to_default_values()
        self.print_func = cast(None, PRINT_STRING_FUN)

        i = 0
        while i < len(argv):
            if argv[i] == "-s":
                i += 1
                self.solver = int(argv[i])
            elif argv[i] == "-k":
                i += 1
                self.kernel = int(argv[i])
            elif argv[i] == "-l":
                i += 1
                self.lambda_ = float(argv[i])
            elif argv[i] == "-m":
                i += 1
                self.mu = float(argv[i])
            elif argv[i] == "-t":
                i += 1
                self.theta = float(argv[i])
            elif argv[i] == "-d":
                i += 1
                self.degree = int(argv[i])
            elif argv[i] == "-g":
                i += 1
                self.gamma = float(argv[i])
            elif argv[i] == "-c":
                i += 1
                self.coef0 = float(argv[i])
            elif argv[i] == "-f":
                i += 1
                self.frequency = int(argv[i])
            elif argv[i] == "-e":
                i += 1
                self.eps = float(argv[i])
            elif argv[i] == "-q":
                self.print_func = PRINT_STRING_FUN(print_null)
            else:
                raise ValueError("Wrong options")
            i += 1

        libodm.set_print_string_function(self.print_func)


class model(Structure):
    _names = ["m", "d", "bias", "param", "w",
              "total_sv", "sv", "sv_coef"]
    _types = [c_int, c_int, c_int, parameter, POINTER(
        c_double), c_int,  POINTER(POINTER(feature_node)), POINTER(c_double)]
    _fields_ = genFields(_names, _types)

    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):  # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
            libodm.free_and_destroy_model(pointer(self))

    def get_w(self):
        w = (c_double * self.d)()
        libodm.get_w(self, w)
        return w[:self.d]


class prediction(Structure):
    _names = ["m", "pre_acc", "pre_label", "pre_value"]
    _types = [c_int, c_double, POINTER(c_double), POINTER(c_double)]
    _fields_ = genFields(_names, _types)

    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):  # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
            libodm.free_and_destroy_prediction(pointer(self))

    def get_pre_label(self):
        pre_label = (c_double * self.m)()
        libodm.get_pre_label(self, pre_label)
        return pre_label[:self.m]

    def get_pre_value(self):
        pre_value = (c_double * self.m)()
        libodm.get_pre_value(self, pre_value)
        return pre_value[:self.m]


def toPyFormat(struct_ptr):  # Convert a ctypes POINTER(struct) to a Python format
    if bool(struct_ptr) == False:
        raise ValueError("Null pointer")
    s = struct_ptr.contents
    s.__createfrom__ = 'C'
    return s


def read_libsvm_format(data_file_name, return_scipy=False):
    # read libsvm format data from data_file_name and return labels y and data instances x
    # return_scipy=False -> [y, x], y: list, x: list of dictionary
    # return_scipy=True -> [y, x], y: ndarray, x: csr_matrix
    if scipy != None and return_scipy:
        prob_y = array('d')
        prob_x = array('d')
        row_ptr = array('l', [0])
        col_idx = array('l')
    else:
        prob_y = []
        prob_x = []
        row_ptr = [0]
        col_idx = []
    indx_start = 1
    for i, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        if len(line) == 1:  # an instance with all zero features
            line += ['']
        label, features = line
        prob_y.append(float(label))
        if scipy != None and return_scipy:
            nz = 0
            for e in features.split():
                ind, val = e.split(":")
                if ind == '0':
                    indx_start = 0
                val = float(val)
                if val != 0:
                    col_idx.append(int(ind)-indx_start)
                    prob_x.append(val)
                    nz += 1
            row_ptr.append(row_ptr[-1]+nz)
        else:
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            prob_x += [xi]
    if scipy != None and return_scipy:
        prob_y = scipy.frombuffer(prob_y, dtype='d')
        prob_x = scipy.frombuffer(prob_x, dtype='d')
        col_idx = scipy.frombuffer(col_idx, dtype='l')
        row_ptr = scipy.frombuffer(row_ptr, dtype='l')
        prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
    return (prob_y, prob_x)


fillprototype(libodm.train, POINTER(model), [
              POINTER(problem), POINTER(parameter)])

fillprototype(libodm.predict, POINTER(prediction), [POINTER(
    problem), POINTER(model)])

fillprototype(libodm.get_w, None, [POINTER(model), POINTER(c_double)])

fillprototype(libodm.get_pre_label, None, [
              POINTER(prediction), POINTER(c_double)])
fillprototype(libodm.get_pre_value, None, [
              POINTER(prediction), POINTER(c_double)])

fillprototype(libodm.check_parameter, c_char_p, [POINTER(parameter)])

fillprototype(libodm.free_and_destroy_model, None, [POINTER(POINTER(model))])
fillprototype(libodm.free_and_destroy_prediction,
              None, [POINTER(POINTER(prediction))])

fillprototype(libodm.set_print_string_function,
              None, [CFUNCTYPE(None, c_char_p)])

fillprototype(libodm.save_model,
              None, [c_char_p, POINTER(model)])
