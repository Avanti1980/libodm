mkdir -p model result

./libodm-train -s 0 -k 0 -l 64 -m 0.4 -t 0.9 -e 0.001 -b data/svmguide1_train model/svmguide1_train.model
./libodm-predict data/svmguide1_test model/svmguide1_train.model result/pre_labels

./libodm-train -s 1 -k 0 -l 128 -m 0.1 -t 0.9 -e 0.001 -b data/svmguide1_train model/svmguide1_train.model
./libodm-predict data/svmguide1_test model/svmguide1_train.model result/pre_labels

./libodm-train -s 2 -k 0 -l 64 -m 0.2 -t 0.8 -e 0.001 -b data/svmguide1_train model/svmguide1_train.model
./libodm-predict data/svmguide1_test model/svmguide1_train.model result/pre_labels

./libodm-train -s 0 -k 2 -l 4096 -m 0.3 -t 0.1 -g 16 -e 0.001 data/svmguide1_train model/svmguide1_train.model
./libodm-predict data/svmguide1_test model/svmguide1_train.model result/pre_labels

rm -rf model result