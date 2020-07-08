#include "mex.h"
#include "odm.h"
#include <stdlib.h>
#include <string.h>

#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

#define NUM_OF_RETURN_FIELD 8

static const char *field_names[] = {
    "m",
    "d",
    "bias",
    "parameters",
    "w",
    "total_sv",
    "sv",
    "sv_coef",
};

const char *c_to_octave(mxArray *plhs[], int num_of_feature, struct model *model) {
    int i, j, out_id = 0;
    double *ptr;
    mxArray *return_model, **rhs = (mxArray **)mxMalloc(sizeof(mxArray *) * NUM_OF_RETURN_FIELD);

    // m
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model->m;
    out_id++;

    // d
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model->d;
    out_id++;

    // bias
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model->bias;
    out_id++;

    // parameters
    rhs[out_id] = mxCreateDoubleMatrix(4, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model->param.kernel;
    ptr[1] = model->param.degree;
    ptr[2] = model->param.gamma;
    ptr[3] = model->param.coef0;
    out_id++;

    // w
    if (model->w != NULL) {
        rhs[out_id] = mxCreateDoubleMatrix(model->d, 1, mxREAL);
        ptr = mxGetPr(rhs[out_id]);
        for (i = 0; i < model->d; i++) {
            ptr[i] = model->w[i];
        }
    } else {
        rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
    }
    out_id++;

    // total_sv
    rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
    ptr = mxGetPr(rhs[out_id]);
    ptr[0] = model->total_sv;
    out_id++;

    // sv
    if (model->sv != NULL) {
        int ir_index, nonzero_element;
        mwIndex *ir, *jc;

        nonzero_element = 0;
        for (i = 0; i < model->total_sv; i++) {
            j = 0;
            while (model->sv[i][j].index != -1) {
                nonzero_element++;
                j++;
            }
        }

        rhs[out_id] = mxCreateSparse(num_of_feature, model->total_sv, nonzero_element, mxREAL);
        ir = mxGetIr(rhs[out_id]);
        jc = mxGetJc(rhs[out_id]);
        ptr = mxGetPr(rhs[out_id]);
        jc[0] = ir_index = 0;
        for (i = 0; i < model->total_sv; i++) {
            int x_index = 0;
            while (model->sv[i][x_index].index != -1) {
                ir[ir_index] = model->sv[i][x_index].index - 1;
                ptr[ir_index] = model->sv[i][x_index].value;
                ir_index++, x_index++;
            }
            jc[i + 1] = jc[i] + x_index;
        }
    } else
        rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
    out_id++;

    // sv_coef
    if (model->sv_coef != NULL) {
        rhs[out_id] = mxCreateDoubleMatrix(model->total_sv, 1, mxREAL);
        ptr = mxGetPr(rhs[out_id]);
        for (i = 0; i < model->total_sv; i++)
            ptr[i] = model->sv_coef[i];
    } else
        rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
    out_id++;

    /* Create a struct matrix contains NUM_OF_RETURN_FIELD fields */
    return_model = mxCreateStructMatrix(1, 1, NUM_OF_RETURN_FIELD, field_names);

    /* Fill struct matrix with input arguments */
    for (i = 0; i < NUM_OF_RETURN_FIELD; i++) {
        mxSetField(return_model, 0, field_names[i], mxDuplicateArray(rhs[i]));
    }

    plhs[0] = return_model;
    mxFree(rhs);
    return NULL;
}

const char *octave_to_c(struct model *model, const mxArray *octave_struct) {
    int i, j, id = 0;
    double *ptr;
    int num_of_fields = mxGetNumberOfFields(octave_struct);
    mxArray **rhs = (mxArray **)mxMalloc(sizeof(mxArray *) * num_of_fields);
    struct feature_node *x_space;

    for (i = 0; i < num_of_fields; i++) {
        rhs[i] = mxGetFieldByNumber(octave_struct, 0, i);
    }

    model->w = NULL;
    model->sv = NULL;
    model->sv_coef = NULL;

    // m
    ptr = mxGetPr(rhs[id]);
    model->m = (int)ptr[0];
    id++;

    // d
    ptr = mxGetPr(rhs[id]);
    model->d = (int)ptr[0];
    id++;

    // bias
    ptr = mxGetPr(rhs[id]);
    model->bias = (int)ptr[0];
    id++;

    // parameters
    ptr = mxGetPr(rhs[id]);
    model->param.kernel = (int)ptr[0];
    model->param.degree = (int)ptr[1];
    model->param.gamma = ptr[2];
    model->param.coef0 = ptr[3];
    id++;

    // w
    if (mxIsEmpty(rhs[id]) == 0) {
        model->w = Malloc(double, model->d);
        ptr = mxGetPr(rhs[id]);
        for (i = 0; i < model->d; i++)
            model->w[i] = ptr[i];
    }
    id++;

    // total_sv
    ptr = mxGetPr(rhs[id]);
    model->total_sv = (int)ptr[0];
    id++;

    // sv
    if (mxIsEmpty(rhs[id]) == 0) {
        int sr, elements;
        int num_samples;
        mwIndex *ir, *jc;

        sr = (int)mxGetN(rhs[id]);

        ptr = mxGetPr(rhs[id]);
        ir = mxGetIr(rhs[id]);
        jc = mxGetJc(rhs[id]);

        num_samples = (int)mxGetNzmax(rhs[id]);

        elements = num_samples + sr;

        model->sv = (struct feature_node **)malloc(sr * sizeof(struct feature_node *));
        x_space = (struct feature_node *)malloc(elements * sizeof(struct feature_node));

        // SV is in column
        for (i = 0; i < sr; i++) {
            int low = (int)jc[i], high = (int)jc[i + 1];
            int x_index = 0;
            model->sv[i] = &x_space[low + i];
            for (j = low; j < high; j++) {
                model->sv[i][x_index].index = (int)ir[j] + 1;
                model->sv[i][x_index].value = ptr[j];
                x_index++;
            }
            model->sv[i][x_index].index = -1;
        }
        id++;
    }

    // sv_coef
    if (mxIsEmpty(rhs[id]) == 0) {
        model->sv_coef = Malloc(double, model->total_sv);
        ptr = mxGetPr(rhs[id]);
        for (i = 0; i < model->total_sv; i++)
            model->sv_coef[i] = ptr[i];
    }
    id++;

    mxFree(rhs);
    return NULL;
}
