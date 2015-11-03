
#include "factorize.h"

void conv_with_fft( float *images, float *filters,  int n_img, int n_channel, int n_filter, int img_h, int img_w, int kernel_size)
{

    float *in_src;
    float *in_kernel;
    
    float *dst;
    float *dst_fft;
    float *dst_dot_sum;

    int FFTW_FACTORS[5] = {7,5,3,2,0}; // end with zero to detect the end of the array

    int w_src = img_w;
    int h_src = img_h;
    int w_kernel = kernel_size;
    int h_kernel = kernel_size;
    int h_fftw = find_closest_factor(h_src + h_kernel - 1,FFTW_FACTORS); 
    int w_fftw = find_closest_factor(w_src + w_kernel - 1,FFTW_FACTORS);   

    // full convolution
    int h_dst = h_src + h_kernel-1;
    int w_dst = w_src + w_kernel-1;

    int h_offset_kernel = (h_fftw-h_kernel)/2;
    int w_offset_kernel = (w_fftw-w_kernel)/2;
    int h_offset_src = (h_fftw-h_src)/2;
    int w_offset_src = (w_fftw-w_src)/2;

    
    // We use CLFFT_COMPLEX_INTERLEAVED, so it stroes as (real_0, imaginary_0), (real_1, imaginary_1), ...
    in_src  = (float *)malloc(sizeof(float) * 2 * h_fftw * w_fftw * n_channel * n_img);
    in_kernel  = (float *)malloc(sizeof(float) * 2 * h_fftw * w_fftw * n_channel * n_filter);
    
    // dst_dot_sum are re-usable
    dst_dot_sum  = (float *)malloc(sizeof(float) * 2 * h_fftw * w_fftw);
    dst_fft  = (float *)malloc(sizeof(float) * 2 * h_fftw * w_fftw);
    dst  = (float *)malloc(sizeof(float) * h_src * w_src * n_filter * n_img);


    memset(in_src, 0, sizeof(float) * 2 * h_fftw * w_fftw * n_channel * n_img);
    memset(in_kernel, 0, sizeof(float) * 2 * h_fftw * w_fftw * n_channel * n_filter);
    memset(dst_dot_sum, 0, sizeof(float) * 2 * h_fftw * w_fftw);
    memset(dst_fft, 0, sizeof(float) * 2 * h_fftw * w_fftw);
    memset(dst, 0, sizeof(float) * h_src * w_src * n_filter * n_img);
 
    // pad image
    for(int g = 0 ; g < n_img ; ++g){
        for(int c = 0 ; c < n_channel ; ++c){
            for(int i = 0 ; i < h_src ; ++i){
                for(int j = 0 ; j < w_src ; ++j){
                    in_src[(g*n_channel*h_fftw*w_fftw+c*h_fftw*w_fftw+(i%h_fftw)*w_fftw+(j%w_fftw))*2+0] = images[g*n_channel*h_src*w_src+c*h_src*w_src+i*w_src+j];
                }
            }
        }
    }


    // pad and flip filter
    for(int f = 0 ; f < n_filter ; ++f){
        for(int c = 0 ; c < n_channel ; ++c){
            for(int i = 0 ; i < h_kernel ; ++i){
                for(int j = 0 ; j < w_kernel ; ++j){
                    // XXX flip the kernel x[i][j]=x[h-1-i][w-1-j]
                    in_kernel[(f*n_channel*h_fftw*w_fftw+c*h_fftw*w_fftw+(i%h_fftw)*w_fftw+(j%w_fftw))*2+0] = filters[f*n_channel*h_kernel*w_kernel+c*h_kernel*w_kernel+(h_kernel-i-1)*w_kernel+(w_kernel-j-1)];
                    
                }
            }
        }
    }
   

    // If use in place FFT, then we do not need out_src and out_kernel, but for simplicity, we use out of place FFT
    cl_mem cl_in_src = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_img, NULL, &errcode);
    cl_mem cl_out_src = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_img, NULL, &errcode);
    cl_mem cl_in_kernel = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_filter, NULL, &errcode);
    cl_mem cl_out_kernel = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_filter, NULL, &errcode);
    cl_mem cl_dst_dot_sum = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw, NULL, &errcode);
    cl_mem cl_dst_fft = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(cl_float2) * h_fftw * w_fftw, NULL, &errcode);

    // copy imgeas and filters to GPU
    errcode = clEnqueueWriteBuffer(clCommandQue,  cl_in_src, CL_TRUE, 0, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_img, in_src, 0, NULL, NULL);
    errcode = clEnqueueWriteBuffer(clCommandQue,  cl_in_kernel, CL_TRUE, 0, sizeof(cl_float2) * h_fftw * w_fftw * n_channel * n_filter, in_kernel, 0, NULL, NULL);

    size_t clLengths_2D[CLFFT_2D] = {h_fftw, w_fftw};

    // setup clFFT
    clfftSetupData fftSetup;
    errcode = clfftInitSetupData(&fftSetup);
    ASSERT_CL_RETURN( errcode );
    errcode = clfftSetup(&fftSetup);
    ASSERT_CL_RETURN( errcode );

    clfftPlanHandle img_planHandle;

    clfftPlanHandle filter_planHandle;

    clfftPlanHandle output_planHandle;

    
    /* Create a default plan for a complex FFT. */
    errcode = clfftCreateDefaultPlan(&img_planHandle, clGPUContext, CLFFT_2D, clLengths_2D);
    errcode = clfftCreateDefaultPlan(&filter_planHandle, clGPUContext, CLFFT_2D, clLengths_2D);
    errcode = clfftCreateDefaultPlan(&output_planHandle, clGPUContext, CLFFT_2D, clLengths_2D);
    
    /* Set plan parameters. using batched FFT */
    errcode = clfftSetPlanBatchSize(img_planHandle, n_channel * n_img);
    errcode = clfftSetPlanPrecision(img_planHandle, CLFFT_SINGLE);
    errcode = clfftSetResultLocation(img_planHandle, CLFFT_OUTOFPLACE);
    
    errcode = clfftSetPlanBatchSize(filter_planHandle, n_channel * n_filter);
    errcode = clfftSetPlanPrecision(filter_planHandle, CLFFT_SINGLE);
    errcode = clfftSetResultLocation(filter_planHandle, CLFFT_OUTOFPLACE);
    
    errcode = clfftSetPlanPrecision(output_planHandle, CLFFT_SINGLE);
    errcode = clfftSetResultLocation(output_planHandle, CLFFT_OUTOFPLACE);
    
    /* Bake the plan. */
    errcode = clfftBakePlan(img_planHandle, 1, &clCommandQue, NULL, NULL);
    errcode = clfftBakePlan(filter_planHandle, 1, &clCommandQue, NULL, NULL);
    errcode = clfftBakePlan(output_planHandle, 1, &clCommandQue, NULL, NULL);


    /* FFT on images */
    errcode = clfftEnqueueTransform(img_planHandle, CLFFT_FORWARD, 1, &clCommandQue, 0, NULL, NULL, &cl_in_src, &cl_out_src, NULL);
    ASSERT_CL_RETURN( errcode );    

    /* FFT on kernel */
    errcode = clfftEnqueueTransform(filter_planHandle, CLFFT_FORWARD, 1, &clCommandQue, 0, NULL, NULL, &cl_in_kernel, &cl_out_kernel, NULL);
    ASSERT_CL_RETURN( errcode );   

    // perfrom dot product on one image and one filter at a time
    for(int g = 0;g<n_img;g++){
        for(int f =0;f<n_filter;f++){
            //printf("Process image %d filter %d\n",g,f);

            int src_offset = g * h_fftw * w_fftw * n_channel;
            int kernel_offset = f * h_fftw * w_fftw * n_channel;
            // dot product and sum the results across channels
            cl_launch_dot_product_and_sum_kernel(cl_out_src, src_offset, cl_out_kernel, kernel_offset, h_fftw, w_fftw, n_channel, cl_dst_dot_sum);

            // Backwards FFT the dot product to get the real number values, the result is full convolution result
            errcode = clfftEnqueueTransform(output_planHandle, CLFFT_BACKWARD, 1, &clCommandQue, 0, NULL, NULL, &cl_dst_dot_sum, &cl_dst_fft, NULL);

            // extract the "same" convolution results from "full" convolution results 
            int dst_offset = g * h_src * w_src * n_filter + f * h_src * w_src;
            int h_offset = h_kernel/2;
            int w_offset = w_kernel/2;            
            errcode = clEnqueueReadBuffer(clCommandQue, cl_dst_fft, CL_TRUE, 0, sizeof(cl_float2) * h_fftw * w_fftw, dst_fft, 0, NULL, NULL);
            printf("Result from image %d filter %d:\n",g,f);
            for(int i = 0 ; i < h_src ; ++i){
                for(int j = 0 ; j < w_src ; ++j){
                    dst[dst_offset+i*w_src+j] = dst_fft[((i+h_offset)*w_fftw+j+w_offset)*2+0];
                    printf("%f ",dst[dst_offset+i*w_src+j]);
                }
                printf("\n");               
            }

        }
    }


    /* Release the plan. */
    errcode = clfftDestroyPlan( &img_planHandle );
    errcode = clfftDestroyPlan( &filter_planHandle );
    errcode = clfftDestroyPlan( &output_planHandle);


/********************************************************************/
/****************** free memory *************************************/


    free(in_src);
    free(in_kernel);
    free(dst);
    free(dst_fft);
    free(dst_dot_sum);

    errcode = clReleaseMemObject(cl_in_src);
    errcode = clReleaseMemObject(cl_out_src);
    errcode = clReleaseMemObject(cl_in_kernel);
    errcode = clReleaseMemObject(cl_out_kernel);
    errcode = clReleaseMemObject(cl_dst_dot_sum);
    errcode = clReleaseMemObject(cl_dst_fft);

    /* Release clFFT library. */
    clfftTeardown( );
	
    return;
}


void cl_launch_dot_product_and_sum_kernel(cl_mem src, int src_offset, cl_mem filter, int filter_offset, int h_fftw, int w_fftw, int n_channel, cl_mem dot_sum)
{
    	
	size_t globalItemSize,localItemSize,shared_size;
	
	localItemSize=256;
	globalItemSize=h_fftw*w_fftw;;
	
    errcode =  clSetKernelArg(dot_product_and_sum_kernel, 0, sizeof(cl_mem), (void *)&src);	
	errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 1, sizeof(int), (void *)&src_offset);
    errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 2, sizeof(cl_mem), (void *)&filter);	
	errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 3, sizeof(int), (void *)&filter_offset);
	errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 4, sizeof(float), (void *)&h_fftw);
	errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 5, sizeof(float), (void *)&w_fftw);
	errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 6, sizeof(float), (void *)&n_channel);
    errcode |=  clSetKernelArg(dot_product_and_sum_kernel, 7, sizeof(cl_mem), (void *)&dot_sum);	

	// Execute the OpenCL kernel
	errcode |= clEnqueueNDRangeKernel(clCommandQue, dot_product_and_sum_kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);
	errcode |= clFinish(clCommandQue);
	ASSERT_CL_RETURN(errcode);
	
}



