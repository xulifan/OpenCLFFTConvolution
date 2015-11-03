#include "main.h"
#include "opencl_init.c"
#include "conv_with_fft.c"

void print_usage()
{
    printf("To run the program: ./main.exe n_img n_channel img_h img_w kernel_size stride n_filter\n");
    printf("For simplicity, stride size should be 1 and img_h == img_w\n");
    exit(0);

}

int main(int argc, char *argv[])
{

    if(argc!=8) print_usage();

    int n_img=atoi(argv[1]);
    int n_channel=atoi(argv[2]);
    int img_h=atoi(argv[3]);
    int img_w=atoi(argv[4]);
    int kernel_size=atoi(argv[5]);
    int stride=atoi(argv[6]);
    int n_filter=atoi(argv[7]);

    // For simplicity, stride size is set to be 1, but we can always extract results from the full convolution results using different stride size
    assert(stride == 1);

    // For simplicity, each image has the same height and width
    assert(img_h == img_w);

    // input images
    float *bottom_data=(float *)calloc(n_img*n_channel*img_h*img_w,sizeof(float));

    // input filters
    float *filter_data=(float *)calloc(n_filter*n_channel*img_h*img_w,sizeof(float));
    
    for(int i=0;i<n_img*n_channel*img_h*img_w;i++){
        bottom_data[i]=((float)rand()/RAND_MAX);
    }

    for(int i=0;i<n_filter*n_channel*kernel_size*kernel_size;i++){
        filter_data[i]=((float)rand()/RAND_MAX);
    }
    
    OpenCL_init();
   
    conv_with_fft( bottom_data, filter_data,  n_img, n_channel, n_filter, img_h, img_w, kernel_size);
    
    cl_clean_up();

    free(bottom_data);
    free(filter_data);

    return 0;
    
}



