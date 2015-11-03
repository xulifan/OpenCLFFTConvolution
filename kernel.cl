__kernel void dot_product_and_sum_kernel(__global const float2 *src, int src_offset,
    __global const float2 *filter, int filter_offset,
    int h_fftw, int w_fftw, int n_channel,
    __global float2 *dot_sum) {
    int tid = get_global_id(0);
    int fftsize = h_fftw * w_fftw;
    if(tid >= fftsize) return;
    float2 temp={0.0f, 0.0f};
    float re_s, im_s, re_k, im_k;
    for(int i =0;i < n_channel; i++ ){
        re_s=src[src_offset+i*fftsize+tid].x;
        im_s=src[src_offset+i*fftsize+tid].y;
        re_k=filter[filter_offset+i*fftsize+tid].x;
        im_k=filter[filter_offset+i*fftsize+tid].y;
        temp.x+=re_s * re_k - im_s * im_k;
        temp.y+=re_s * im_k + im_s * re_k;
        //temp+=src[src_offset+i*fftsize+tid] * filter[filter_offset+i*fftsize+tid];
    }
    dot_sum[tid]=temp;
}
