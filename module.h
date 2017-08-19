/*
* Alessandro Maragno - 09/11/2016
* module.h
*/

#ifndef MODULE_H
#define MODULE_H

#include <stdarg.h>

#include "tensor.h"

void set_input(tensor *in, module *m);
tensor get_output(module *m);

module init_module(module_t type, int n_args, ...);
void free_module(module *m);
void print_module(module m, int print_tensors);
void forward(module *m);

void add_module(module *container, module to_add);

#endif