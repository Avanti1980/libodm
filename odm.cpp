#include "odm.h"
#include "tron.h"
#include <algorithm>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using std::random_shuffle;
const char *libodm_version = LIBODM_VERSION;
template <class T>
static inline void swap(T &x, T &y) {
    T t = x;
    x = y;
    y = t;
}
#ifndef min
template <class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }
#endif
#ifndef max
template <class T>
static inline T max(T x, T y) { return (x > y) ? x : y; }
#endif
#define INF HUGE_VAL
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

static void print_string_stdout(const char *s) {
    fputs(s, stdout);
    fflush(stdout);
}

static void (*libodm_print_string)(const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt, ...) {
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    (*libodm_print_string)(buf);
}
#else
static void info(const char *fmt, ...) {}
#endif

class sparse_operator {
public:
    static double nrm2_sq(const feature_node *x) { // |x|^2
        double ret = 0;
        while (x->index != -1) {
            ret += x->value * x->value;
            x++;
        }
        return ret;
    }

    static double dot(const double *s, const feature_node *x) { // <s, x>
        double ret = 0;
        while (x->index != -1) {
            ret += s[x->index - 1] * x->value;
            x++;
        }
        return ret;
    }

    static void axpy(const double a, const feature_node *x, double *y) { // y = a x + y
        while (x->index != -1) {
            y[x->index - 1] += a * x->value;
            x++;
        }
    }
};

double dot(const struct feature_node *px, const struct feature_node *py) {
    double sum = 0;
    while (px->index != -1 && py->index != -1) {
        if (px->index == py->index) {
            sum += px->value * py->value;
            ++px;
            ++py;
        } else if (px->index > py->index)
            ++py;
        else
            ++px;
    }
    return sum;
}

static void solve_cd(const problem *prob, const parameter *param, model *model_) {
    int m = prob->m, d = prob->d;
    double lambda = param->lambda, mu = param->mu, theta = param->theta, eps = param->eps;
    int i, j, s, iter = 0, max_iter = 1000, kernel = param->kernel;

    int m_double = 2 * m;
    int active_size = m_double;
    int *index = new int[m_double];
    double *alpha = new double[m_double];

    double G;
    double coef1 = 0.5 * m * (1 - theta) * (1 - theta) / lambda;
    double coef2 = coef1 / mu;

    // PG: projected gradient for shrinking and stopping
    double PG, PGmax_new, PGmin_new;
    double PGmax_old = INF;
    double PGmin_old = -INF;

    // initialize index = i and alpha = 0
    for (i = 0; i < m_double; i++) {
        alpha[i] = 0;
        index[i] = i;
    }

    double *x_square = new double[m];
    for (i = 0; i < m; i++)
        x_square[i] = sparse_operator::nrm2_sq(prob->x[i]);

    double *Q = NULL;

    // initialize w = 0 for linear kernel and kernel matrix Q for nonlinear kernel
    if (kernel == LINEAR)
        for (i = 0; i < d; i++)
            model_->w[i] = 0;
    else {
        Q = new double[m * m]; // Q = Y X^T X Y
        switch (kernel) {
            case POLY:
                info("dual coordinate descent, polynomial kernel\n");
                for (i = 0; i < m; i++)
                    for (j = i; j < m; j++)
                        Q[i * m + j] = prob->y[i] * prob->y[j] * pow(param->gamma * dot(prob->x[i], prob->x[j]) + param->coef0, param->degree);
                break;
            case RBF:
                info("dual coordinate descent, rbf kernel\n");
                for (i = 0; i < m; i++)
                    for (j = i; j < m; j++)
                        Q[i * m + j] = prob->y[i] * prob->y[j] * exp(-param->gamma * (x_square[i] + x_square[j] - 2 * dot(prob->x[i], prob->x[j])));
                break;
            case SIGMOID:
                info("dual coordinate descent, sigmoid kernel\n");
                for (i = 0; i < m; i++)
                    for (j = i; j < m; j++)
                        Q[i * m + j] = prob->y[i] * prob->y[j] * tanh(param->gamma * dot(prob->x[i], prob->x[j]) + param->coef0);
                break;
        }
        for (i = 0; i < m; i++)
            for (j = 0; j < i; j++)
                Q[i * m + j] = Q[j * m + i];

        if (prob->bias == 1)
            info("An additional all one feature is appended to the feature matrix\n");
    }

    while (iter < max_iter) {
        PGmax_new = -INF;
        PGmin_new = INF;

        for (i = 0; i < active_size; i++) {
            j = i + rand() % (active_size - i);
            swap(index[i], index[j]);
        }

        for (s = 0; s < active_size; s++) {
            i = index[s];
            G = 0;

            // calculate gradient
            if (i < m) {
                if (kernel == LINEAR) {
                    G += sparse_operator::dot(model_->w, prob->x[i]);
                    G *= prob->y[i];
                } else
                    for (j = 0; j < m; j++)
                        G += Q[i * m + j] * (alpha[j] - alpha[m + j]);
                G += coef1 * alpha[i];
                G += (theta - 1);
            } else {
                if (kernel == LINEAR) {
                    G -= sparse_operator::dot(model_->w, prob->x[i - m]);
                    G *= prob->y[i - m];
                } else {
                    for (j = 0; j < m; j++) {
                        G += Q[(i - m) * m + j] * (-alpha[j] + alpha[m + j]);
                    }
                }
                G += coef2 * alpha[i];
                G += (theta + 1);
            }

            PG = 0;
            if (alpha[i] == 0) {
                if (G > PGmax_old) {
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                } else if (G < 0)
                    PG = G;
            } else
                PG = G;

            PGmax_new = max(PGmax_new, PG);
            PGmin_new = min(PGmin_new, PG);

            // update solution w = X Y (alpha[1:m] - alpha[m+1:2m])
            if (fabs(PG) > 1.0e-12) {
                if (i < m) {
                    if (kernel == LINEAR) {
                        double alpha_old = alpha[i];
                        alpha[i] = max(alpha[i] - PG / (x_square[i] + coef1), 0.0);
                        double inc = (alpha[i] - alpha_old) * prob->y[i];
                        feature_node *xi = prob->x[i];
                        while (xi->index != -1) {
                            model_->w[xi->index - 1] += inc * xi->value;
                            xi++;
                        }
                    } else
                        alpha[i] = max(alpha[i] - PG / (Q[i * (m + 1)] + coef1), 0.0);
                } else {
                    if (kernel == LINEAR) {
                        double alpha_old = alpha[i];
                        alpha[i] = max(alpha[i] - PG / (x_square[i - m] + coef2), 0.0);
                        double inc = (alpha_old - alpha[i]) * prob->y[i - m];
                        feature_node *xi = prob->x[i - m];
                        while (xi->index != -1) {
                            model_->w[xi->index - 1] += inc * xi->value;
                            xi++;
                        }

                    } else
                        alpha[i] = max(alpha[i] - PG / (Q[(i - m) * (m + 1)] + coef2), 0.0);
                }
            }
        }

        iter++;

        if (iter % 10 == 0) info(".");

        if (PGmax_new - PGmin_new <= eps) {
            if (active_size == m_double)
                break;
            else {
                active_size = m_double;
                info("*");
                PGmax_old = INF;
                PGmin_old = -INF;
                continue;
            }
        }
        PGmax_old = PGmax_new;
        PGmin_old = PGmin_new;
        if (PGmax_old <= 0) {
            PGmax_old = INF;
        }
        if (PGmin_old >= 0) {
            PGmin_old = -INF;
        }
    }

    info("\noptimization finished, #iter = %d\n", iter);

    if (kernel != LINEAR) {
        delete[] Q;
        model_->w = NULL;
        model_->total_sv = 0;

        bool *nonzero = new bool[m];
        for (i = 0; i < m; i++) {
            if (fabs(alpha[i] - alpha[m + i]) > 0) {
                model_->total_sv++;
                nonzero[i] = true;
            } else
                nonzero[i] = false;
        }

        model_->sv = Malloc(feature_node *, model_->total_sv);
        model_->sv_coef = Malloc(double, model_->total_sv);

        j = 0;
        for (i = 0; i < m; i++) {
            if (nonzero[i]) {
                model_->sv[j] = prob->x[i];
                model_->sv_coef[j] = prob->y[i] * (alpha[i] - alpha[m + i]);
                j++;
            }
        }
        delete[] nonzero;
    } else if (iter >= max_iter)
        info("WARNING: reaching max number of iterations! Using -s 1 may be faster!\n");

    delete[] alpha;
    delete[] index;
    delete[] x_square;
}

class odm_tr : public function {
public:
    odm_tr(const problem *prob, const parameter *param);
    ~odm_tr();
    double fun(double *w);           // calculate the objective function value
    void grad(double *w, double *g); // calculate the gradient
    void Hv(double *s, double *Hs);
    int get_nr_variable(void);

protected:
    void Xv(double *v, double *Xv);
    void subXv(double *v, double *Xv, int flag);
    void subXTv(double *v, double *XTv, int flag);
    double *z; // desicion value of all the examples
    double *z1;
    double *z2;
    int *I1;    // index set of examples: y_i w' x_i < 1 - theta
    int *I2;    // index set of examples: y_i w' x_i > 1 + theta
    int sizeI1; // |I1|
    int sizeI2; // |I2|
    const problem *prob;
    const parameter *param;
    double coef1, coef2;
};

odm_tr::odm_tr(const problem *prob, const parameter *param) {
    this->prob = prob;
    this->param = param;
    z = new double[prob->m];
    z1 = new double[prob->m];
    z2 = new double[prob->m];
    I1 = new int[prob->m];
    I2 = new int[prob->m];
    coef1 = param->lambda / (prob->m * (1 - param->theta) * (1 - param->theta));
    coef2 = coef1 * param->mu;
}

odm_tr::~odm_tr() {
    delete[] z;
    delete[] z1;
    delete[] z2;
    delete[] I1;
    delete[] I2;
}

double odm_tr::fun(double *w) {
    double f = 0;
    double l1 = 0;
    double l2 = 0;

    for (int i = 0; i < prob->d; i++)
        f += w[i] * w[i];
    f /= 2.0;

    Xv(w, z); // calculate the desicion values z = X * w

    for (int i = 0; i < prob->m; i++) {
        z[i] *= prob->y[i];
        double d = z[i] - (1 - param->theta);
        if (d < 0) // if y_i w' x_i < 1 - theta
            l1 += d * d;
        d = z[i] - (1 + param->theta);
        if (d > 0) // if y_i w' x_i > 1 + theta
            l2 += d * d;
    }

    f += coef1 * l1 + coef2 * l2;
    return f;
}

void odm_tr::grad(double *w, double *g) {
    double *g1 = new double[prob->d];
    double *g2 = new double[prob->d];
    sizeI1 = 0;
    sizeI2 = 0;

    for (int i = 0; i < prob->m; i++) {
        if (z[i] < 1 - param->theta) {
            z1[sizeI1] = prob->y[i] * (z[i] - 1 + param->theta);
            I1[sizeI1] = i;
            sizeI1++;
        } else if (z[i] > 1 + param->theta) {
            z2[sizeI2] = prob->y[i] * (z[i] - 1 - param->theta);
            I2[sizeI2] = i;
            sizeI2++;
        }

        subXTv(z1, g1, 1);
        subXTv(z2, g2, 2);

        for (int i = 0; i < prob->d; i++)
            g[i] = w[i] + 2 * (coef1 * g1[i] + coef2 * g2[i]);
    }

    delete[] g1;
    delete[] g2;
}

int odm_tr::get_nr_variable(void) { return prob->d; }

void odm_tr::Hv(double *s, double *Hs) {
    double *wa1 = new double[sizeI1];
    double *wa2 = new double[sizeI2];
    double *Hs1 = new double[prob->d];
    double *Hs2 = new double[prob->d];

    subXv(s, wa1, 1);
    subXv(s, wa2, 2);
    subXTv(wa1, Hs1, 1);
    subXTv(wa2, Hs2, 2);

    for (int i = 0; i < prob->d; i++)
        Hs[i] = s[i] + 2 * (coef1 * Hs1[i] + coef2 * Hs2[i]);

    delete[] wa1;
    delete[] wa2;
    delete[] Hs1;
    delete[] Hs2;
}

void odm_tr::Xv(double *v, double *Xv) {
    for (int i = 0; i < prob->m; i++)
        Xv[i] = sparse_operator::dot(v, prob->x[i]);
}

void odm_tr::subXv(double *v, double *Xv, int l) {
    if (l == 1)
        for (int i = 0; i < sizeI1; i++)
            Xv[i] = sparse_operator::dot(v, prob->x[I1[i]]);
    else if (l == 2)
        for (int i = 0; i < sizeI2; i++)
            Xv[i] = sparse_operator::dot(v, prob->x[I2[i]]);
}

void odm_tr::subXTv(double *v, double *XTv, int l) {
    for (int i = 0; i < prob->d; i++)
        XTv[i] = 0;

    if (l == 1)
        for (int i = 0; i < sizeI1; i++) {
            feature_node *s = prob->x[I1[i]];
            while (s->index != -1) {
                XTv[s->index - 1] += v[i] * s->value;
                s++;
            }
        }
    else if (l == 2)
        for (int i = 0; i < sizeI2; i++) {
            feature_node *s = prob->x[I2[i]];
            while (s->index != -1) {
                XTv[s->index - 1] += v[i] * s->value;
                s++;
            }
        }
}

class svrg {
public:
    svrg(int d, double lambda, double mu, double theta, double eta, double eps, int frequency, double *w);
    ~svrg();
    void reset(double *p);
    double gradient_norm();
    void renorm();
    double dloss(double decision_value, double y);
    void full_gradient(const problem *prob); // calculate the full gradient
    void train_svrg_one(const struct feature_node *x, const double y, double eta);
    void train_svrg(const struct problem *prob);

private:
    int d;
    double lambda;
    double mu;
    double theta;
    double eta; // initial step size
    double eps;
    int frequency;
    double *w;
    double *w_last; // w of last iteration
    double *g;      // full gradient
    double *g_last; // full gradient of last iteration
    double coef1;
    double coef2;
    double *w_last_minus_g;
    double a;
};

svrg::svrg(int d, double lambda, double mu, double theta, double eta, double eps, int frequency, double *w)
    : d(d), lambda(lambda), mu(mu), theta(theta), eta(eta), eps(eps), frequency(frequency), w(w), a(1.0) {
    w_last = new double[d];
    g = new double[d];
    g_last = new double[d];
    w_last_minus_g = new double[d];
    coef1 = lambda / ((1 - theta) * (1 - theta));
    coef2 = coef1 * mu;
}

svrg::~svrg() {
    delete[] w_last;
    delete[] g;
    delete[] g_last;
    delete[] w_last_minus_g;
}

void svrg::reset(double *p) {
    for (int i = 0; i < d; i++)
        p[i] = 0;
}

double svrg::gradient_norm() {
    double norm = 0;
    for (int i = 0; i < d; i++)
        norm += g[i] * g[i];
    return norm;
}

void svrg::renorm() {
    for (int i = 0; i < d; i++)
        w[i] /= a;
    a = 1.0;
}

double svrg::dloss(double decision_value, double y) {
    double z = decision_value * y;
    if (z < 1 - theta)
        return 2 * coef1 * y * (z + theta - 1);
    if (z > 1 + theta)
        return 2 * coef2 * y * (z - theta - 1);
    return 0;
}

void svrg::full_gradient(const problem *prob) {
    reset(g);

    // g = w + dloss * x
    for (int i = 0; i < prob->m; i++) {
        double decision_value = sparse_operator::dot(w, prob->x[i]);
        sparse_operator::axpy(dloss(decision_value, prob->y[i]), prob->x[i], g);
    }
    for (int i = 0; i < d; i++)
        g[i] = w[i] + g[i] / prob->m;
}

/*
w = w - eta * (w + dloss(w) * x - w_last - dloss(w_last) * x + g)
  = (1 - eta) * w + eta * (w_last - g) - eta * dloss(w) * x + eta * dloss(w_last) * x
  = (1 - eta) * w + eta * (w_last - g) - eta * (dloss(w) - dloss(w_last)) * x
*/
void svrg::train_svrg_one(const feature_node *x, const double y, double eta) {
    double decision_value = sparse_operator::dot(w, x) / a;
    double inc = -dloss(decision_value, y); // - dloss(w)
    decision_value = sparse_operator::dot(w_last, x);
    inc += dloss(decision_value, y); // dloss(w_last)

    a = a / (1 - eta);
    inc *= eta * a;

    for (int i = 0; i < d; i++)
        w[i] += eta * a * w_last_minus_g[i]; // + eta * (w_last - g)

    if (inc != 0) sparse_operator::axpy(inc, x, w);

    if (a > 1e5) renorm();
}

void svrg::train_svrg(const problem *prob) {
    int iter_max = 100;
    int *index = new int[prob->m];

    reset(w);            // initial w as zero vector
    full_gradient(prob); // calculate the full gradient
    memcpy(g_last, g, sizeof(double) * d);
    memcpy(w_last, w, sizeof(double) * d);
    for (int i = 0; i < d; i++)
        w_last_minus_g[i] = w_last[i] - g[i];

    double gnorm = gradient_norm();
    int iter = 0;
    while (gnorm > eps) {
        if (iter == iter_max) {
            info("Warning: reach the maximum iteration number\n");
            break;
        }
        iter++;

        // update frequency * m iteration
        for (int j = 0; j < frequency; j++) {
            for (int i = 0; i < prob->m; i++)
                index[i] = i;
            random_shuffle(index, index + prob->m);
            for (int i = 0; i < prob->m; i++)
                train_svrg_one(prob->x[index[i]], prob->y[index[i]], eta);
        }
        renorm();

        full_gradient(prob); // update full gradient

        gnorm = gradient_norm();
        info("iter %2d  |g|^2 %5.3e  stepsize %5.3e\n", iter, gnorm, eta);

        // calculate the latest step size eta = |w - w_last|^2 / (w - w_last)' (g - g_last) (frequency * m)
        if (iter > 1) {
            eta = 0;
            double wg = 0;
            for (int i = 0; i < d; i++) {
                eta += (w[i] - w_last[i]) * (w[i] - w_last[i]); // |w - w_last|^2
                wg += (w[i] - w_last[i]) * (g[i] - g_last[i]);  // (w - w_last)' (g - g_last)
            }
            eta /= (wg * frequency * prob->m);
        }

        // save w_last and g_last
        memcpy(w_last, w, sizeof(double) * d);
        memcpy(g_last, g, sizeof(double) * d);

        // update w_last - g
        for (int i = 0; i < d; i++)
            w_last_minus_g[i] = w_last[i] - g[i];
    }
    delete[] index;
}

static void solve_svrg(const problem *prob, const parameter *param, double *w) {
    double eta = 1.0 / max(prob->m, prob->d); // initial step size
    svrg svrg_odm(prob->d, param->lambda, param->mu, param->theta, eta, param->eps, param->frequency, w);
    svrg_odm.train_svrg(prob);
}

model *train(const problem *prob, const parameter *param) {
    model *model_ = Malloc(model, 1);
    model_->m = prob->m;
    model_->d = prob->d;
    model_->bias = prob->bias;
    model_->param = *param;

    info("----------------------------------------------------------------------------------------\n");
    if (prob->bias == 1)
        info("Train: %d instances, %d features, ", prob->m, prob->d - 1);
    else
        info("Train: %d instances, %d features, ", prob->m, prob->d);

    if (param->kernel != LINEAR)
        solve_cd(prob, param, model_);
    else {
        model_->total_sv = -1;
        model_->sv = NULL;
        model_->sv_coef = NULL;
        model_->w = Malloc(double, prob->d);
        if (param->solver == CD) {
            info("dual coordinate descent, linear kernel\n");
            if (prob->bias == 1)
                info("An additional all one feature is appended to the feature matrix\n");
            solve_cd(prob, param, model_);
        } else if (param->solver == TR) {
            info("trust region Newton method, linear kernel\n");
            if (prob->bias == 1)
                info("An additional all one feature is appended to the feature matrix\n");
            function *fun_obj = NULL;
            fun_obj = new odm_tr(prob, param);
            TRON tron_obj(fun_obj, param->eps / max(prob->m, prob->d));
            tron_obj.set_print_string(libodm_print_string);
            tron_obj.tron(model_->w);
            delete fun_obj;
        } else {
            info("svrg, linear kernel\n");
            if (prob->bias == 1)
                info("An additional all one feature is appended to the feature matrix\n");
            solve_svrg(prob, param, model_->w);
        }
    }
    return model_;
}

prediction *predict(const problem *prob, const model *model_) {
    prediction *prediction_ = Malloc(prediction, 1);
    prediction_->m = prob->m;
    prediction_->pre_label = Malloc(double, prob->m);
    prediction_->pre_value = Malloc(double, prob->m);

    int i, j, correct = 0;
    double *x_square = NULL;

    if (model_->param.kernel == RBF) {
        x_square = new double[model_->total_sv];
        for (i = 0; i < model_->total_sv; i++)
            x_square[i] = sparse_operator::nrm2_sq(model_->sv[i]);
    }

    for (i = 0; i < prob->m; i++) {
        if (model_->param.kernel == LINEAR)
            prediction_->pre_value[i] = sparse_operator::dot(model_->w, prob->x[i]);
        else {
            prediction_->pre_value[i] = 0;
            for (j = 0; j < model_->total_sv; j++)
                switch (model_->param.kernel) {
                    case POLY:
                        prediction_->pre_value[i] += model_->sv_coef[j] * pow(model_->param.gamma * dot(prob->x[i], model_->sv[j]) + model_->param.coef0, model_->param.degree);
                        break;
                    case RBF:
                        prediction_->pre_value[i] += model_->sv_coef[j] * exp(-model_->param.gamma * (sparse_operator::nrm2_sq(prob->x[i]) + x_square[j] - 2 * dot(prob->x[i], model_->sv[j])));
                        break;
                    case SIGMOID:
                        prediction_->pre_value[i] += model_->sv_coef[j] * tanh(model_->param.gamma * dot(prob->x[i], model_->sv[j]) + model_->param.coef0);
                        break;
                }
        }

        prediction_->pre_label[i] = (prediction_->pre_value[i] > 0) ? 1 : -1;

        if (prediction_->pre_label[i] == prob->y[i])
            correct++;
    }

    prediction_->pre_acc = (double)correct / prob->m;
    info("Test: acc = %g%% (%d/%d)\n", prediction_->pre_acc * 100, correct, prob->m);

    if (model_->param.kernel == RBF)
        delete[] x_square;

    return prediction_;
}

void get_w(const model *model_, double *w) {
    if (model_->w != NULL)
        for (int i = 0; i < model_->d; i++)
            w[i] = model_->w[i];
}

void get_pre_label(const prediction *prediction_, double *pre_label) {
    if (prediction_->pre_label != NULL)
        for (int i = 0; i < prediction_->m; i++)
            pre_label[i] = prediction_->pre_label[i];
}

void get_pre_value(const prediction *prediction_, double *pre_value) {
    if (prediction_->pre_value != NULL)
        for (int i = 0; i < prediction_->m; i++)
            pre_value[i] = prediction_->pre_value[i];
}

const char *check_parameter(const parameter *param) {
    if (param->eps <= 0)
        return "eps <= 0";
    if (param->mu < 0 || param->mu > 1)
        return "mu < 0 or mu > 1";
    if (param->theta < 0 || param->theta >= 1)
        return "theta < 0 || theta >= 1";
    if (param->solver != CD && param->solver != TR && param->solver != SVRG)
        return "unknown solver type";
    if (param->kernel != LINEAR && param->kernel != POLY && param->kernel != RBF && param->kernel != SIGMOID)
        return "unknown kernel type";
    return NULL;
}

void free_and_destroy_model(struct model **model_ptr_ptr) {
    struct model *model_ptr = *model_ptr_ptr;
    if (model_ptr != NULL) {
        if (model_ptr->w != NULL)
            free(model_ptr->w);
        if (model_ptr->sv != NULL)
            free(model_ptr->sv);
        if (model_ptr->sv_coef != NULL)
            free(model_ptr->sv_coef);
        free(model_ptr);
    }
}

void free_and_destroy_prediction(struct prediction **prediction_ptr_ptr) {
    struct prediction *prediction_ptr = *prediction_ptr_ptr;
    if (prediction_ptr != NULL) {
        if (prediction_ptr->pre_label != NULL)
            free(prediction_ptr->pre_label);
        if (prediction_ptr->pre_value != NULL)
            free(prediction_ptr->pre_value);
        free(prediction_ptr);
    }
}

void set_print_string_function(void (*print_func)(const char *)) {
    if (print_func == NULL)
        libodm_print_string = &print_string_stdout;
    else
        libodm_print_string = print_func;
}

int save_model(const char *model_file_name, const struct model *model_) {
    FILE *fp = fopen(model_file_name, "w");
    if (fp == NULL) return -1;

    fprintf(fp, "instance %d\n", model_->m);
    fprintf(fp, "feature %d\n", model_->d);
    fprintf(fp, "bias %d\n", model_->bias);

    const parameter &param = model_->param;
    fprintf(fp, "solver %d\n", param.solver);
    fprintf(fp, "kernel %d\n", param.kernel);

    if (param.kernel == POLY)
        fprintf(fp, "degree %d\n", param.degree);

    if (param.kernel == POLY || param.kernel == RBF || param.kernel == SIGMOID)
        fprintf(fp, "gamma %g\n", param.gamma);

    if (param.kernel == POLY || param.kernel == SIGMOID)
        fprintf(fp, "coef0 %g\n", param.coef0);

    if (param.kernel == LINEAR) {
        fprintf(fp, "w\n");
        for (int i = 0; i < model_->d; i++)
            fprintf(fp, "%.16g ", model_->w[i]);
    } else {
        fprintf(fp, "total_sv %d\n", model_->total_sv);
        fprintf(fp, "sv\n");
        for (int i = 0; i < model_->total_sv; i++) {
            fprintf(fp, "%.16g ", model_->sv_coef[i]);
            const feature_node *p = model_->sv[i];
            while (p->index != -1) {
                fprintf(fp, "%d:%.16g ", p->index, p->value);
                p++;
            }
            fprintf(fp, "\n");
        }
    }

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return -1;
    else
        return 0;
}

static char *line = NULL;
static int max_line_len;

static char *readline(FILE *input) {
    int len;

    if (fgets(line, max_line_len, input) == NULL)
        return NULL;

    while (strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        line = (char *)realloc(line, max_line_len);
        len = (int)strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL)
            break;
    }
    return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)                        \
    do {                                                      \
        if (fscanf(_stream, _format, _var) != 1) return NULL; \
    } while (0)

struct model *load_model(const char *model_file_name) {
    FILE *fp = fopen(model_file_name, "r");
    if (fp == NULL) return NULL;

    model *model_ = Malloc(model, 1);
    parameter &param = model_->param;
    model_->w = NULL;
    model_->sv = NULL;
    model_->sv_coef = NULL;

    char cmd[81];
    while (1) {
        FSCANF(fp, "%80s", cmd);
        if (strcmp(cmd, "instance") == 0)
            FSCANF(fp, "%d", &model_->m);
        else if (strcmp(cmd, "feature") == 0)
            FSCANF(fp, "%d", &model_->d);
        else if (strcmp(cmd, "bias") == 0)
            FSCANF(fp, "%d", &model_->bias);
        else if (strcmp(cmd, "solver") == 0)
            FSCANF(fp, "%d", &param.solver);
        else if (strcmp(cmd, "kernel") == 0)
            FSCANF(fp, "%d", &param.kernel);
        else if (strcmp(cmd, "degree") == 0)
            FSCANF(fp, "%d", &param.degree);
        else if (strcmp(cmd, "gamma") == 0)
            FSCANF(fp, "%lf", &param.gamma);
        else if (strcmp(cmd, "coef0") == 0)
            FSCANF(fp, "%lf", &param.coef0);
        else if (strcmp(cmd, "total_sv") == 0)
            FSCANF(fp, "%d", &model_->total_sv);
        else if (strcmp(cmd, "w") == 0) {
            model_->w = Malloc(double, model_->d);
            for (int i = 0; i < model_->d; i++)
                FSCANF(fp, "%lf", &model_->w[i]);
            break;
        } else if (strcmp(cmd, "sv") == 0) {
            while (1) {
                int c = getc(fp);
                if (c == EOF || c == '\n') break;
            }
            break;
        }
    }

    // read sv_coef and SV
    if (param.kernel != LINEAR) {
        int elements = 0;
        long pos = ftell(fp);

        max_line_len = 1024;
        line = Malloc(char, max_line_len);
        char *p, *endptr, *idx, *val;

        while (readline(fp) != NULL) {
            p = strtok(line, ":");
            while (1) {
                p = strtok(NULL, ":");
                if (p == NULL)
                    break;
                ++elements;
            }
        }

        elements += model_->total_sv;

        fseek(fp, pos, SEEK_SET);

        model_->sv_coef = Malloc(double, model_->total_sv);
        model_->sv = Malloc(feature_node *, model_->total_sv);
        feature_node *x_space = Malloc(feature_node, elements);

        int i, j = 0;
        for (i = 0; i < model_->total_sv; i++) {
            readline(fp);
            model_->sv[i] = &x_space[j];

            p = strtok(line, " \t");
            model_->sv_coef[i] = strtod(p, &endptr);

            while (1) {
                idx = strtok(NULL, ":");
                val = strtok(NULL, " \t");

                if (val == NULL)
                    break;
                x_space[j].index = (int)strtol(idx, &endptr, 10);
                x_space[j].value = strtod(val, &endptr);

                ++j;
            }
            x_space[j++].index = -1;
        }
        free(line);
    }

    if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

    return model_;
}