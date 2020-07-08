#include "odm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mex.h"
#include "odm_model.h"

#define CMD_LEN 2048

#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

int print_null(const char *s, ...) {}

static void fake_answer(int nlhs, mxArray *plhs[]) {
    for (int i = 0; i < nlhs; i++) {
        plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
}

void do_predict(int nlhs, mxArray *plhs[], const mxArray *prhs[], struct model *model_) {
    int i, j, k, low, high, correct = 0;
    int testing_instance_num, dim;
    double *groundtruth, *label, *samples, *x_square; // read data
    double *pre_acc, *pre_label, *pre_value;          // prediction
    mwIndex *ir, *jc;
    struct feature_node **x_test, **x_train;
    struct feature_node *x_space_test, *x_space_train;

    groundtruth = mxGetPr(prhs[0]);         // prhs[0] = groundtruth
    testing_instance_num = mxGetN(prhs[1]); // prhs[1] = testing instance matrix
    dim = mxGetM(prhs[1]);

    if (mxGetM(prhs[0]) != testing_instance_num) {
        mexPrintf("length of label vector does not match # of instances.\n");
        fake_answer(nlhs, plhs);
        return;
    }

    // read testing data
    samples = mxGetPr(prhs[1]);
    ir = mxGetIr(prhs[1]);
    jc = mxGetJc(prhs[1]);
    x_test = Malloc(struct feature_node *, testing_instance_num);
    if (model_->bias == 1) {
        x_space_test = Malloc(struct feature_node, mxGetNzmax(prhs[1]) + 2 * testing_instance_num);
    } else {
        x_space_test = Malloc(struct feature_node, mxGetNzmax(prhs[1]) + testing_instance_num);
    }

    j = 0;
    for (i = 0; i < testing_instance_num; i++) {
        x_test[i] = &x_space_test[j];
        low = (int)jc[i], high = (int)jc[i + 1];
        for (k = low; k < high; k++) {
            x_space_test[j].index = (int)ir[k] + 1;
            x_space_test[j].value = samples[k];
            j++;
        }
        if (model_->bias == 1) {
            x_space_test[j].index = dim + 1;
            x_space_test[j].value = 1;
            j++;
        }
        x_space_test[j++].index = -1;
    }

    if (model_->param.kernel == RBF) {
        x_square = Malloc(double, model_->total_sv);
        for (i = 0; i < model_->total_sv; i++) {
            x_square[i] = dot(model_->sv[i], model_->sv[i]);
        }
    }

    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    pre_acc = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(testing_instance_num, 1, mxREAL);
    pre_label = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(testing_instance_num, 1, mxREAL);
    pre_value = mxGetPr(plhs[2]);

    for (i = 0; i < testing_instance_num; i++) {
        pre_value[i] = 0;

        if (model_->param.kernel == LINEAR) {
            while (x_test[i]->index != -1) {
                pre_value[i] += model_->w[x_test[i]->index - 1] * x_test[i]->value;
                x_test[i]++;
            }
        } else {
            for (j = 0; j < model_->total_sv; j++) {
                switch (model_->param.kernel) {
                    case POLY:
                        pre_value[i] += model_->sv_coef[j] * pow(model_->param.gamma * dot(x_test[i], model_->sv[j]) + model_->param.coef0, model_->param.degree);
                        break;
                    case RBF:
                        pre_value[i] += model_->sv_coef[j] * exp(-model_->param.gamma * (dot(x_test[i], x_test[i]) + x_square[j] - 2 * dot(x_test[i], model_->sv[j])));
                        break;
                    case SIGMOID:
                        pre_value[i] += model_->sv_coef[j] * tanh(model_->param.gamma * dot(x_test[i], model_->sv[j]) + model_->param.coef0);
                        break;
                }
            }
        }

        pre_label[i] = (pre_value[i] > 0) ? 1 : -1;
        if (pre_label[i] == groundtruth[i]) {
            correct++;
        }
    }

    mexPrintf("Test: acc = %g%% (%d/%d)\n", (double)correct / testing_instance_num * 100, correct, testing_instance_num);

    pre_acc[0] = (double)correct / testing_instance_num * 100;

    free(x_test);
    free(x_space_test);
    if (model_->param.kernel != LINEAR) {
        free(x_train);
        free(x_space_train);
        if (model_->param.kernel == RBF) {
            free(x_square);
        }
    }
}

void exit_with_help() {
    mexPrintf("Usage: [accuracy, predicted_label, decision_values] = predict(testing_label_vector, testing_instance_matrix, training_label_vector, training_instance_matrix, model)\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    struct model *model_;
    char cmd[CMD_LEN];

    if (nrhs != 3) {
        exit_with_help();
        fake_answer(nlhs, plhs);
        return;
    }

    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
        mexPrintf("error: label vector and instance matrix must be double\n");
        fake_answer(nlhs, plhs);
        return;
    }

    if (mxIsStruct(prhs[2])) {
        const char *error_msg;

        model_ = Malloc(struct model, 1);
        error_msg = octave_to_c(model_, prhs[2]);
        if (error_msg) {
            mexPrintf("error: can't read model: %s\n", error_msg);
            free_and_destroy_model(&model_);
            fake_answer(nlhs, plhs);
            return;
        }

        if (mxIsSparse(prhs[1]))
            do_predict(nlhs, plhs, prhs, model_);
        else {
            mexPrintf("instance_matrix must be sparse, use sparse(instance_matrix) first\n");
            fake_answer(nlhs, plhs);
        }

        free_and_destroy_model(&model_);
    } else {
        mexPrintf("model file should be a struct array\n");
        fake_answer(nlhs, plhs);
    }

    return;
}
