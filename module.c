/*
* Alessandro Maragno - 09/11/2016
* module.c
*/

#include <stdlib.h>
#include <stdio.h>

#include "convolutional_module.h"
#include "pool_module.h"
#include "activation_module.h"
#include "container_module.h"

void set_input(tensor *in, module *m){ m->input = in; }
tensor get_output(module *m){ return m->output; }

module init_module(module_t type, int n_args, ...)
{
	module new_mod = {0};
	
	va_list arg_list;	
	va_start(arg_list, n_args);
	
	switch(type)
	{
		case CONVOLUTIONAL:
		{
			int n_fil = va_arg(arg_list, int);
			int ker_h = va_arg(arg_list, int);
			int ker_w = va_arg(arg_list, int);
			int pad_h = va_arg(arg_list, int);
			int pad_w = va_arg(arg_list, int);
			int stride_h = va_arg(arg_list, int);
			int stride_w = va_arg(arg_list, int);
			int input_depth = va_arg(arg_list, int);
			tensor_data_t *weights = va_arg(arg_list, tensor_data_t *);
			tensor_data_t *bias = va_arg(arg_list, tensor_data_t *);
			
			new_mod = init_convolutional_mod(n_fil, ker_h, ker_w, pad_h, pad_w, stride_h, stride_w, input_depth, weights, bias);
			new_mod.type = type;
			
			break;
		}
		case CONTAINER:
		{
			container_t cont_type = va_arg(arg_list, container_t);
			int n_modules = va_arg(arg_list, int);
			int concat_dim = va_arg(arg_list, int);
			
			new_mod = init_container_mod(cont_type, n_modules, concat_dim);
			new_mod.type = type;
			
			break;
		}
		case POOL:
		{
			int ker_h = va_arg(arg_list, int);
			int ker_w = va_arg(arg_list, int);
			int pad_h = va_arg(arg_list, int);
			int pad_w = va_arg(arg_list, int);
			int stride_h = va_arg(arg_list, int);
			int stride_w = va_arg(arg_list, int);
			
			new_mod = init_pool_mod(ker_h, ker_w, pad_h, pad_w, stride_h, stride_w);
			new_mod.type = type;
						
			break;
		}	
		case ACTIVATION:
		{
			activation_t act_type = va_arg(arg_list, activation_t);
			
			new_mod = init_activation_mod(act_type);
			new_mod.type = type;
			
			break;
		}	
		default:
			break;			
	}
	
	va_end(arg_list);
	
	return new_mod;
}

void forward(module *m)
{
	switch(m->type)
	{
		case CONVOLUTIONAL:
			forward_convolutional_mod(m);
			break;
			
		case CONTAINER:
			forward_container_mod(m);
			break;
			
		case POOL:
			forward_pool_mod(m);
			break;
			
		case ACTIVATION:
			forward_activation_mod(m);
			break;
			
		default:
			break;
	}
}

void free_module(module *m)
{
	if(!m) return;
	
	int i;
	
	if(m->n_modules > 0 && m->cont_type != CONCAT)
	{
		module *modules = m->modules;
		
		for(i = 0; i < m->n_modules; i++)
			free_tensor(modules[i].input);
	}
		
	if(m->filters)
	{
		for(i = 0; i < m->n_filters; i++) free_tensor(&(m->filters[i]));
		free(m->filters);
	}
		
	for(i = 0; i < m->n_modules; i++)
		free_module(&(m->modules[i]));
	
	free(m->modules);
	
	if(m->type == CONTAINER) free_tensor(&(m->output));
}

void print_module(module m, int print_tensors)
{
	switch(m.type)
	{
		case CONVOLUTIONAL:
			print_convolutional_mod(m, print_tensors);
			break;
			
		case CONTAINER:
			print_container_mod(m, print_tensors);
			break;
			
		case POOL:
		{
			print_pool_mod(m, print_tensors);
			break;
		}	
		case ACTIVATION:
			print_activation_mod(m, print_tensors);
			break;
			
		default:
			break;
	}
}

void add_module(module *container, module to_add)
{
	if(container->type != CONTAINER)
	{
		fprintf(stderr, "ERROR: module type must be CONTAINER to add modules to it. Returning.\n");
		return;
	}
	
	add_module_to_cont(container, to_add);
}