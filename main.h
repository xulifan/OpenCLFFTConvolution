#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <CL/cl.h>
#include <clBLAS.h>
#include <clFFT.h>

#define ASSERT_CL_RETURN( ret )\
   if( (ret) != CL_SUCCESS )\
   {\
      fprintf( stderr, "%s:%d: error: %d %s\n", \
             __FILE__, __LINE__, ret, getOpenCLErrorString( (ret) ));\
      exit(-1);\
   }


//OpenCL variables
#define MAX_SOURCE_SIZE (0x100000)
char str_temp[1024];
char driver_version[1024];
char device_version[1024];
char device_extension[1024];
char *source_str;
size_t source_size;
size_t printf_buffer_size;
cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_device_local_mem_type local_mem_type;	
cl_ulong global_mem_size;
cl_ulong global_mem_cache_size;
cl_ulong global_mem_cacheline_size;
cl_ulong max_mem_alloc_size;
cl_ulong local_mem_size;
cl_kernel dot_product_and_sum_kernel;
int DEVICE_QUERY = 0;

//opencl init
void OpenCL_init();
void read_cl_file();
void cl_initialization();
void cl_load_prog();
void cl_clean_up();
const char *getOpenCLErrorString(cl_int err);


void cl_launch_dot_product_and_sum_kernel(cl_mem src, int src_offset, cl_mem filter, int filter_offset, int h_fftw, int w_fftw, int n_channel, cl_mem dot_sum);

void conv_with_fft( float *images, float *filters,  int n_img, int n_channel, int n_filter, int img_h, int img_w, int kernel_size);


