#include "odm.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

void print_null(const char *s) {}

void exit_with_help() {
    printf(
        "Usage: train [options] training_set_file [model_file]\n"
        "options:\n"
        "libodm_options:\n"
        "-s solver    : default 0\n"
        "               0 -- dual coordinate descent (dual)\n"
        "	           1 -- trust region Newton method (primal)\n"
        "	           2 -- svrg (primal)\n"
        "-k kernel    : default 0\n"
        "	             0 -- linear\n"
        "	             1 -- polynomial\n"
        "	             2 -- rbf\n"
        "	             3 -- sigmoid\n"
        "-l lambda    : odm parameter \n"
        "-m mu        : odm parameter \n"
        "-t theta     : odm parameter \n"
        "-d degree    : polynomial kernel parameter \n"
        "-g gamma     : polynomial / rbf / sigmoid kernel parameter \n"
        "-c coef0     : polynomial / sigmoid kernel parameter \n"
        "-f frequency : scan data times for svrg\n"
        "-e epsilon   : set the epsilon in optimization\n"
        "-q           : quiet mode\n");
    exit(1);
}

void exit_input_error(int line_num) {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
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

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model *model_;

int main(int argc, char **argv) {
    char input_file_name[1024];
    char model_file_name[1024];
    const char *error_msg;

    parse_command_line(argc, argv, input_file_name, model_file_name);
    read_problem(input_file_name);
    error_msg = check_parameter(&param);

    if (error_msg) {
        fprintf(stderr, "ERROR: %s\n", error_msg);
        exit(1);
    }

    model_ = train(&prob, &param);
    if (save_model(model_file_name, model_)) {
        fprintf(stderr, "can't save model to file %s\n", model_file_name);
        exit(1);
    }
    free_and_destroy_model(&model_);

    free(prob.y);
    free(prob.x);
    free(x_space);
    free(line);

    return 0;
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name) {
    int i;
    void (*print_func)(const char *) = NULL; // default printing to stdout

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

    // parse options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') break;
        ++i;
        if (i >= argc && argv[i - 1][1] != 'b' && argv[i - 1][1] != 'q') // since options -b and -q have no parameter
            exit_with_help();
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
                fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                exit_with_help();
                break;
        }
    }

    set_print_string_function(print_func);

    if (param.solver != CD) { // TR and SVRG are only used for linear kernel
        param.kernel = LINEAR;
    }

    // determine filenames
    if (i >= argc)
        exit_with_help();

    strcpy(input_file_name, argv[i]);

    if (i < argc - 1)
        strcpy(model_file_name, argv[i + 1]);
    else {
        char *p = strrchr(argv[i], '/');
        if (p == NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name, "%s.model", p);
    }
}

// read in a problem (in libsvm format)
void read_problem(const char *filename) {
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;

    if (fp == NULL) {
        fprintf(stderr, "can't open input file %s\n", filename);
        exit(1);
    }

    prob.m = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char, max_line_len);
    while (readline(fp) != NULL) {
        char *p = strtok(line, " \t"); // label

        // features
        while (1) {
            p = strtok(NULL, " \t");
            if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        prob.m++;
    }
    rewind(fp);

    prob.y = Malloc(double, prob.m);
    prob.x = Malloc(struct feature_node *, prob.m);
    x_space = Malloc(struct feature_node, elements + prob.m);

    max_index = 0;
    j = 0;
    for (i = 0; i < prob.m; i++) {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(i + 1);

        prob.y[i] = strtod(label, &endptr);
        if (endptr == label || *endptr != '\0')
            exit_input_error(i + 1);

        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL)
                break;

            errno = 0;
            x_space[j].index = (int)strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i + 1);
            else
                inst_max_index = x_space[j].index;

            errno = 0;
            x_space[j].value = strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i + 1);

            ++j;
        }

        if (inst_max_index > max_index)
            max_index = inst_max_index;

        if (prob.bias == 1)
            x_space[j++].value = prob.bias;

        x_space[j++].index = -1;
    }

    if (prob.bias == 1) {
        prob.d = max_index + 1;
        for (i = 1; i < prob.m; i++)
            (prob.x[i] - 2)->index = prob.d;
        x_space[j - 2].index = prob.d;
    } else
        prob.d = max_index;

    fclose(fp);
}
