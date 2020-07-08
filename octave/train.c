#include "odm.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mex.h"
#include "odm_model.h"

#define CMD_LEN 2048
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

void print_null(const char *s) {}
void print_string_octave(const char *s) { mexPrintf(s); }

void exit_with_help() {
    mexPrintf("Usage: model = train(training_label_vector, training_instance_matrix, 'libodm_options');\n"
              "libodm_options:\n"
              "   -s solver  : default 0\n"
              "                0 -- dual coordinate descent (dual)\n"
              "   	           1 -- trust region Newton method (primal)\n"
              "   	           2 -- svrg (primal)\n"
              "   -k kernel  : default 0\n"
              "   	           0 -- linear\n"
              "   	           1 -- polynomial\n"
              "   	           2 -- rbf\n"
              "   	           3 -- sigmoid\n"
              "   -l lambda  : odm parameter \n"
              "   -m mu      : odm parameter \n"
              "   -t theta   : odm parameter \n"
              "   -d degree  : polynomial kernel parameter \n"
              "   -g gamma   : polynomial / rbf / sigmoid kernel parameter \n"
              "   -c coef0   : polynomial / sigmoid kernel parameter \n"
              "   -f frequency   : scan data times for svrg\n"
              "   -e epsilon : set the epsilon in optimization\n"
              "   -q         : quiet mode\n");
}

struct parameter param; // set by parse_command_line
struct problem prob;    // set by read_problem
struct model *model_;
struct feature_node *x_space;

int parse_command_line(int nrhs, const mxArray *prhs[]) {
    int i, argc = 1;
    char cmd[CMD_LEN];
    char *argv[CMD_LEN / 2];
    void (*print_func)(const char *) = print_string_octave; // default printing to octave display

    // default values
    param.solver = CD;
    param.kernel = LINEAR;
    param.lambda = 1;
    param.mu = 0.8;
    param.theta = 0.2;
    param.degree = 2;
    param.gamma = 0.1;
    param.coef0 = 0.5;
    param.frequency = 5;
    param.eps = 0.01;
    prob.bias = 0;

    if (nrhs <= 1) {
        return 1;
    }

    // put options in argv[]
    if (nrhs == 3) {
        mxGetString(prhs[2], cmd, mxGetN(prhs[2]) + 1);
        if ((argv[argc] = strtok(cmd, " ")) != NULL) {
            while ((argv[++argc] = strtok(NULL, " ")) != NULL) {
                ;
            }
        }
    }

    // parse options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') break;
        ++i;
        if (i >= argc && argv[i - 1][1] != 'b' && argv[i - 1][1] != 'q') // since options -b and -q have no parameter
            return 1;

        switch (argv[i - 1][1]) {
            case 's':
                param.solver = atoi(argv[i]);
                break;
            case 'k':
                param.kernel = atoi(argv[i]);
                break;
            case 'l':
                param.lambda = atof(argv[i]);
                break;
            case 'm':
                param.mu = atof(argv[i]);
                break;
            case 't':
                param.theta = atof(argv[i]);
                break;
            case 'd':
                param.degree = atoi(argv[i]);
                break;
            case 'g':
                param.gamma = atof(argv[i]);
                break;
            case 'c':
                param.coef0 = atof(argv[i]);
                break;
            case 'f':
                param.frequency = atoi(argv[i]);
                break;
            case 'e':
                param.eps = atof(argv[i]);
                break;
            case 'b':
                prob.bias = 1;
                i--;
                break;
            case 'q':
                print_func = &print_null;
                i--;
                break;
            default:
                mexPrintf("unknown option\n");
                return 1;
        }
    }

    set_print_string_function(print_func);

    if (param.solver != CD) { // TR and SVRG are only used for linear kernel
        param.kernel = LINEAR;
    }

    return 0;
}

static void fake_answer(int nlhs, mxArray *plhs[]) {
    for (int i = 0; i < nlhs; i++) {
        plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
}

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat) {
    mwIndex *ir, *jc;
    int i, j, low, high, k;
    double *samples, *labels;

    prob.x = NULL;
    prob.y = NULL;
    x_space = NULL;

    prob.d = mxGetM(instance_mat); // dimension
    prob.m = mxGetN(instance_mat); // instance number

    if (mxGetM(label_vec) != prob.m) {
        mexPrintf("length of label vector does not match # of instances.\n");
        return -1;
    }

    // each column is one instance
    labels = mxGetPr(label_vec);
    samples = mxGetPr(instance_mat);
    ir = mxGetIr(instance_mat);
    jc = mxGetJc(instance_mat);

    prob.y = Malloc(double, prob.m);
    prob.x = Malloc(struct feature_node *, prob.m);
    if (prob.bias == 1) {
        x_space = Malloc(struct feature_node, mxGetNzmax(instance_mat) + 2 * prob.m);
    } else {
        x_space = Malloc(struct feature_node, mxGetNzmax(instance_mat) + prob.m);
    }

    j = 0;
    for (i = 0; i < prob.m; i++) {
        prob.x[i] = &x_space[j];
        prob.y[i] = labels[i];
        low = jc[i], high = jc[i + 1];
        for (k = low; k < high; k++) {
            x_space[j].index = (int)ir[k] + 1;
            x_space[j].value = samples[k];
            j++;
        }
        if (prob.bias == 1) {
            x_space[j].index = prob.d + 1;
            x_space[j].value = 1;
            j++;
        }
        x_space[j++].index = -1;
    }
    if (prob.bias == 1)
        prob.d++;

    return 0;
}

// Interface function of octave
// prhs[0]: label, prhs[1]: (column sparse) instance matrix , prhs[2]: options
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    const char *error_msg;
    int d = (int)mxGetM(prhs[1]);

    if (nlhs > 1) {
        exit_with_help();
        fake_answer(nlhs, plhs);
        return;
    }

    if (nrhs == 2 || nrhs == 3) {
        int err = 0;

        if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
            mexPrintf("error: label vector and instance matrix must be double\n");
            fake_answer(nlhs, plhs);
            return;
        }

        if (mxIsSparse(prhs[0])) {
            mexPrintf("error: label vector should not be in sparse format\n");
            fake_answer(nlhs, plhs);
            return;
        }

        if (parse_command_line(nrhs, prhs)) {
            exit_with_help();
            fake_answer(nlhs, plhs);
            return;
        }

        if (mxIsSparse(prhs[1]))
            err = read_problem_sparse(prhs[0], prhs[1]);
        else {
            mexPrintf("instance_matrix must be column sparse\n");
            fake_answer(nlhs, plhs);
            return;
        }

        // train's original code
        error_msg = check_parameter(&param);

        if (err || error_msg) {
            if (error_msg != NULL)
                mexPrintf("error: %s\n", error_msg);
            free(prob.y);
            free(prob.x);
            free(x_space);
            fake_answer(nlhs, plhs);
            return;
        }

        model_ = train(&prob, &param);

        error_msg = c_to_octave(plhs, d, model_);

        if (error_msg)
            mexPrintf("error: can't convert libodm model to octave matrix: %s\n", error_msg);
        free_and_destroy_model(&model_);

        free(prob.y);
        free(prob.x);
        free(x_space);
    } else {
        exit_with_help();
        fake_answer(nlhs, plhs);
        return;
    }
}
