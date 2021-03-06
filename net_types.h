/*
* Alessandro Maragno - 14/11/2016
* net_types.h
*/

#ifndef NET_TYPES_H
#define NET_TYPES_H

#define N_INIT_ARG_CONV 10
#define N_INIT_ARG_POOL 6
#define N_INIT_ARG_ACTIV 1
#define N_INIT_ARG_CONT 3

#define DEPTH 1
#define HEIGHT 2
#define WIDTH 3

//#define DATA_FORMAT_STR "%lf"
//#define DATA_FORMAT_STR_SP "%lf "
#define DATA_FORMAT_STR "%f"
#define DATA_FORMAT_STR_SP "%f "

//typedef double tensor_data_t;
typedef float tensor_data_t;

typedef long unsigned int address_t;

long unsigned int _mem_alloc_by_model;
long unsigned int _mem_alloc_by_volumes;

typedef struct
{
	int d;
	int h;
	int w;
	tensor_data_t *data;
	address_t data_addr;
} tensor;

typedef enum
{
	IDENTITY,
	BIN_STEP,
	SIGMOID,
	TANH,
	ARCTAN,
	SOFTSIGN,
	RELU,
	PRELU,
	ELU,
	SIN
} activation_t;

typedef enum
{
	CONVOLUTIONAL = 1,
	CONTAINER,
	POOL,
	ACTIVATION
} module_t;

typedef enum
{
	SEQUENTIAL, // one module's output goes to the next module's input
	CONCAT // each module share the same input and their output is concatenated along a selected dimension
	// EXTEND TYPES
} container_t;

typedef struct module
{	// BASICS
	module_t type;
	
	tensor *input; // this points to the output of the previous module, or to the network input 
	tensor output;
	
	// CONVOLUTIONAL
	int n_filters;
	tensor *filters;
	tensor bias;
	
	int pad_h, pad_w;
	int stride_h, stride_w;
	
	// POOL
	int ker_h, ker_w;
	
	// ACTIVATION
	activation_t act_type;
	
	// CONTAINER
	container_t cont_type;
	int concat_dim; // only if cont_type = CONCAT; concat_dim = 0 in case of other container types
	int n_modules;
	struct module *modules;
} module;

typedef struct layer_config {
  int n_layer; //id layer
  int in_w;  // input dimensions
  int in_h;
  int out_w;
  int out_h;
  int in_ch;
  int out_ch;
  int ker_w;
  int ker_h;
  int ker_ch;
  int str_w;
  int str_h;
  int pad_w;
  int pad_h;
  int relu;
  int has_bias;
  unsigned int mem_addr_input;
  unsigned int mem_addr_output;
  unsigned int mem_addr_weights;
} layer_config;

#endif
