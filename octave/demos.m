addpath('../windows');
load('../data/svmguide1_sparse.mat');

% -s 0: dual coordinate descent
model = train(y_train', X_train', '-s 0 -k 0 -l 64 -m 0.4 -t 0.9 -e 0.01 -b'); 
[acc, pre_label, pre_value] = predict(y_test', X_test', model);

% -s 1: trust region Newton method
model = train(y_train', X_train', '-s 1 -k 0 -l 128 -m 0.1 -t 0.9 -e 0.01 -b');
[acc, pre_label, pre_value] = predict(y_test', X_test', model);

% -s 2: svrg
model = train(y_train', X_train', '-s 2 -k 0 -l 64 -m 0.2 -t 0.8 -e 0.01 -b');
[acc, pre_label, pre_value] = predict(y_test', X_test', model);

% -s 0 -k 2: dual coordinate descent with rbf kernel
model = train(y_train', X_train', '-s 0 -k 2 -l 4096 -m 0.3 -t 0.1 -g 16 -e 0.01 -b'); 
[acc, pre_label, pre_value] = predict(y_test', X_test', model);
