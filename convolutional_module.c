/*
* Alessandro Maragno - 09/11/2016
* convolutional_module.c
*/

#include <stdio.h>
#include <stdlib.h>

#include "convolutional_module.h"

module init_convolutional_mod(	int n_fil, 
								int ker_h, 
								int ker_w, 
								int pad_h, 
								int pad_w, 
								int stride_h, 
								int stride_w,
								int input_depth,
								tensor_data_t *weights,
								tensor_data_t *bias	)
{
	module new_conv_mod = {0};
	
	new_conv_mod.stride_h = stride_h < 1 ? 1 : stride_h; // default is 1
	new_conv_mod.stride_w = stride_w < 1 ? 1 : stride_w; // default is 1
	
	new_conv_mod.pad_h = pad_h < 0 ? 0 : pad_h; // default is 0
	new_conv_mod.pad_w = pad_w < 0 ? 0 : pad_w; // default is 0
	
	new_conv_mod.ker_h = ker_h;
	new_conv_mod.ker_w = ker_w;
	
	new_conv_mod.filters = calloc(n_fil, sizeof(tensor));
	
	// TEST
	_mem_alloc_by_model += n_fil*sizeof(tensor);
	
	int fil;
	if(weights)
	{
		tensor_data_t *weights_ptr = weights;
		int i;
		
		for(fil = 0; fil < n_fil; fil++)
		{
			new_conv_mod.filters[fil] = init_tensor(input_depth, ker_w, ker_h, 0);
			// TEST
			_mem_alloc_by_model += input_depth*ker_w*ker_h*sizeof(tensor_data_t);
			
			for(i = 0; i < input_depth*ker_w*ker_h; i++)
			{
				new_conv_mod.filters[fil].data[i] = *weights_ptr;
				
				weights_ptr++;
			}
		}
		
		free(weights);
	}
	else for(fil = 0; fil < n_fil; fil++) new_conv_mod.filters[fil] = init_tensor(input_depth, ker_w, ker_h, 1);
	
	if(bias)
	{
		new_conv_mod.bias.d = n_fil;
		new_conv_mod.bias.w = 1;
		new_conv_mod.bias.h = 1;
		
		// TEST
		_mem_alloc_by_model += n_fil*sizeof(tensor_data_t);
		
		new_conv_mod.bias.data = bias; 
	} 
	else new_conv_mod.bias = init_tensor(n_fil, 1, 1, 1);
	
	new_conv_mod.n_filters = n_fil;
	
	return new_conv_mod;
}

void inline forward_convolutional_mod(module *cm)
{
	cm->output = convolve_tensors(cm->input[0], cm->filters, &(cm->bias), cm->n_filters, cm->pad_h, cm->pad_w, cm->stride_h, cm->stride_w);
}

void print_convolutional_mod(module cm, int print_tensors)
{
	printf("CONVOLUTIONAL MODULE:\n");
	printf("[%d] filters with kernel size: %dx%dx%d\n", cm.n_filters, cm.filters[0].d, cm.ker_w, cm.ker_h);
	printf("bias with size: %dx%dx%d\n", cm.bias.d, cm.bias.w, cm.bias.h);
	printf("Applying horizontal stride of %d and vertical stride of %d\n", cm.stride_w, cm.stride_h);
	
	if(!cm.pad_h && !cm.pad_w) printf("No zero padding\n");
	else printf("Zero padding: horizontal = %d, vertical = %d\n", cm.pad_w, cm.pad_h);
	
	if(print_tensors)
	{
		printf("Input:\n"); 
		if(cm.input)
			print_tensor(cm.input[0]);
		else printf("Input of size\t%dx%dx%d\n", 0, 0, 0);
		printf("\n");
		
		int i;
		for(i = 0; i < cm.n_filters; i++)
		{
			printf("Filter #%d:\n", i);
			print_tensor(cm.filters[i]);
			printf("\n");
		}
		
		printf("Bias:\n");
		print_tensor(cm.bias);
		
		printf("Output:\n"); print_tensor(cm.output);
	}
	else
	{
		if(cm.input)
			printf("input of size\t%dx%dx%d\n", cm.input[0].d, cm.input[0].w, cm.input[0].h);
		else printf("Input of size\t%dx%dx%d\n", 0, 0, 0);
		
		printf("output of size\t%dx%dx%d\n", cm.output.d, cm.output.w, cm.output.h);
	}
}