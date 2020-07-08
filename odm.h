#ifndef _LIBODM_H
#define _LIBODM_H

#define LIBODM_VERSION "200520"

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node {
    int index;
    double value;
};

struct problem {
    int m, d, bias;
    double *y;
    struct feature_node **x;
};

enum { CD,
       TR,
       SVRG }; // solver type

enum { LINEAR,
       POLY,
       RBF,
       SIGMOID }; // kernel type

struct parameter {
    int solver;
    int kernel;
    double lambda, mu, theta;
    int degree;    // for poly
    double gamma;  // for poly / rbf / sigmoid
    double coef0;  // for poly / sigmoid
    int frequency; // for SVRG
    double eps;
};

struct model {
    int m, d, bias;
    struct parameter param;
    double *w;
    int total_sv;
    struct feature_node **sv;
    double *sv_coef;
};

struct prediction {
    int m;
    double pre_acc;
    double *pre_label;
    double *pre_value;
};

double dot(const struct feature_node *px, const struct feature_node *py);

struct model *train(const struct problem *prob, const struct parameter *param);
struct prediction *predict(const struct problem *prob_test, const struct model *model_);

void get_w(const struct model *model_, double *w);

void get_pre_label(const struct prediction *prediction_, double *pre_label);
void get_pre_value(const struct prediction *prediction_, double *pre_value);

void free_and_destroy_model(struct model **model_ptr_ptr);
void free_and_destroy_prediction(struct prediction **prediction_ptr_ptr);

const char *check_parameter(const struct parameter *param);

void set_print_string_function(void (*print_func)(const char *));

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

#ifdef __cplusplus
}
#endif

#endif
