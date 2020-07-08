#include "odm.h"
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

int print_null(const char *s, ...) { return 0; }

static int (*info)(const char *fmt, ...) = &printf;

struct feature_node *x_space;
struct problem prob;
struct model *model_;
struct prediction *pre;

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

void exit_with_help() {
    printf(
        "Usage: predict [options] test_file model_file output_file\n"
        "options:\n"
        "-q : quiet mode (no outputs)\n");
    exit(1);
}

int main(int argc, char **argv) {
    FILE *input, *output;
    int i, j;

    // parse options
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') break;
        ++i;
        switch (argv[i - 1][1]) {
            case 'q':
                info = &print_null;
                i--;
                break;
            default:
                fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                exit_with_help();
                break;
        }
    }

    if (i >= argc)
        exit_with_help();

    input = fopen(argv[i], "r");
    if (input == NULL) {
        fprintf(stderr, "can't open input file %s\n", argv[i]);
        exit(1);
    }

    output = fopen(argv[i + 2], "w");
    if (output == NULL) {
        fprintf(stderr, "can't open output file %s\n", argv[i + 2]);
        exit(1);
    }

    if ((model_ = load_model(argv[i + 1])) == 0) {
        fprintf(stderr, "can't open model file %s\n", argv[i + 1]);
        exit(1);
    }

    int d;
    if (model_->bias == 1)
        d = model_->d - 1;
    else
        d = model_->d;

    int nonzero_element = 0;
    prob.m = 0;
    max_line_len = 1024;
    line = (char *)malloc(max_line_len * sizeof(char));
    while (readline(input) != NULL) {
        char *idx, *val, *label, *endptr;
        int inst_max_index = 0; // strtol gives 0 if wrong format

        label = strtok(line, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(prob.m + 1);

        strtod(label, &endptr);
        if (endptr == label || *endptr != '\0')
            exit_input_error(prob.m + 1);

        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if (val == NULL) break;
            errno = 0;
            int index = (int)strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || index <= inst_max_index)
                exit_input_error(prob.m + 1);
            else
                inst_max_index = index;

            errno = 0;
            strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(prob.m + 1);

            // feature indices larger than those in training are not used
            if (index <= d)
                nonzero_element++;
        }
        prob.m++;
    }

    prob.d = model_->d;
    prob.bias = model_->bias;
    prob.y = Malloc(double, prob.m);
    prob.x = Malloc(struct feature_node *, prob.m);
    if (prob.bias == 1) {
        x_space = Malloc(struct feature_node, nonzero_element + 2 * prob.m);
    } else {
        x_space = Malloc(struct feature_node, nonzero_element + prob.m);
    }

    rewind(input);
    i = j = 0;
    while (readline(input) != NULL) {
        prob.x[i] = &x_space[j];

        char *idx, *val, *label, *endptr;

        label = strtok(line, " \t\n");
        prob.y[i] = strtod(label, &endptr);

        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");
            if (val == NULL) break;

            int index = (int)strtol(idx, &endptr, 10);

            if (index <= d) {
                x_space[j].index = index;
                x_space[j++].value = strtod(val, &endptr);
            }
        }
        if (model_->bias == 1) {
            x_space[j].index = model_->d;
            x_space[j++].value = 1;
        }
        x_space[j++].index = -1;
        i++;
    }

    pre = predict(&prob, model_);

    fprintf(output, "Acc: %.17g\n", pre->pre_acc);
    fprintf(output, "Prediction:\n");
    for (i = 0; i < prob.m; i++)
        fprintf(output, "%d, %.17g\n", (int)pre->pre_label[i], pre->pre_value[i]);

    free_and_destroy_model(&model_);
    free_and_destroy_prediction(&pre);
    free(prob.y);
    free(prob.x);
    free(x_space);
    free(line);

    fclose(input);
    fclose(output);
    return 0;
}
