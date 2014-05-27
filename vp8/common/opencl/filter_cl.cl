#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

#define BILINEAR_ROW_SIZE 2
constant int bilinear_filters[8*2] = {
    128, 0,
    112, 16,
    96, 32,
    80, 48,
    64, 64,
    48, 80,
    32, 96,
    16, 112
};

#define SIXTAP_ROW_SIZE 8
constant short8 sub_pel_filters[8] = {
    //These were originally 8x6, but are padded for vector ops
    {0, 0, 128, 0, 0, 0, 0, 0}, /* note that 1/8 pel positions are just as per alpha -0.5 bicubic */
    {0, -6, 123, 12, -1, 0, 0, 0},
    {2, -11, 108, 36, -8, 1, 0, 0}, /* New 1/4 pel 6 tap filter */
    {0, -9, 93, 50, -6, 0, 0, 0},
    {3, -16, 77, 77, -16, 3, 0, 0}, /* New 1/2 pel 6 tap filter */
    {0, -6, 50, 93, -9, 0, 0, 0},
    {1, -8, 36, 108, -11, 2, 0, 0}, /* New 1/4 pel 6 tap filter */
    {0, -1, 12, 123, -6, 0, 0, 0}
};

__inline short8 sub_pel_filter_values(int offset, constant short8 *sub_pel_filters){
#if 0
    return sub_pel_filters[offset]; //doesn't work on AMD in Windows...?
#else
    short8 result = 0;
    switch (offset){
        case 0:
            result.s2 = 128;
            break;
        case 1:
            result.s1 = -6;
            result.s2 = 123;
            result.s3 = 12;
            result.s4 = -1;
            break;
        case 2:
            result.s0 = 2;
            result.s1 = -11;
            result.s2 = 108;
            result.s3 = 36;
            result.s4 = -8;
            result.s5 = 1;
            break;
        case 3:
            result.s1 = -9; 
            result.s2 = 93;
            result.s3 = 50;
            result.s4 = -6;
            break;
        case 4:
             result.s0 = 3;
             result.s1 = -16;
             result.s2 = 77;
             result.s3 = 77;
             result.s4 = -16;
             result.s5 = 3;
             break;
        case 5:
            result.s1 = -6;
            result.s2 = 50;
            result.s3 = 93;
            result.s4 = -9;
            break;
        case 6:
            result.s0 = 1;
            result.s1 = -8;
            result.s2 = 36;
            result.s3 = 108;
            result.s4 = -11;
            result.s5 = 2;
            break;
        case 7:
            result.s1 = -1;
            result.s2 = 12;
            result.s3 = 123;
            result.s4 = -6;
            break;
        default:
            break;
    }
    return result;
#endif
}

kernel void vp8_filter_block2d_first_pass_kernel(
    __global unsigned char *src_base,
    int src_offset,
    __global int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
){
    uint tid = get_global_id(0);

    global unsigned char *src_ptr = &src_base[src_offset];
    //Note that src_offset will be reset later, which is why we use it now

    int Temp;

    short8 vp8_filter = sub_pel_filter_values(filter_offset, sub_pel_filters);

    if (tid < (output_width*output_height)){
        src_offset = tid + (tid/output_width * (src_pixels_per_line - output_width));

        Temp = (int)(src_ptr[src_offset - 2] * vp8_filter.s0) +
           (int)(src_ptr[src_offset - 1] * vp8_filter.s1) +
           (int)(src_ptr[src_offset]     * vp8_filter.s2) +
           (int)(src_ptr[src_offset + 1] * vp8_filter.s3) +
           (int)(src_ptr[src_offset + 2] * vp8_filter.s4) +
           (int)(src_ptr[src_offset + 3] * vp8_filter.s5) +
           (VP8_FILTER_WEIGHT >> 1);      /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;

        if (Temp < 0)
            Temp = 0;
        else if ( Temp > 255 )
            Temp = 255;

        output_ptr[tid] = Temp;
    }

}

kernel void vp8_filter_block2d_second_pass_kernel
(
    __global int *src_base,
    int src_offset,
    __global unsigned char *output_base,
    int output_offset,
    int output_pitch,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
) {

    uint i = get_global_id(0);

    global int *src_ptr = &src_base[src_offset];
    global unsigned char *output_ptr = &output_base[output_offset];

    int out_offset; //Not same as output_offset...
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    unsigned int src_increment = src_pixels_per_line - output_width;

    short8 vp8_filter = sub_pel_filter_values(filter_offset, sub_pel_filters);
    
    
    if (i < (output_width * output_height)){
        out_offset = i/output_width;
        src_offset = out_offset;

        src_offset = i + (src_offset * src_increment);
        out_offset = i%output_width + (out_offset * output_pitch);

        /* Apply filter */
        Temp = ((int)src_ptr[src_offset - PS2] * vp8_filter.s0 +
           ((int)src_ptr[src_offset -(int)pixel_step] * vp8_filter.s1) +
           ((int)src_ptr[src_offset]                  * vp8_filter.s2) +
           ((int)src_ptr[src_offset + pixel_step]     * vp8_filter.s3) +
           ((int)src_ptr[src_offset + PS2]       * vp8_filter.s4) +
           ((int)src_ptr[src_offset + PS3]       * vp8_filter.s5) +
           (VP8_FILTER_WEIGHT >> 1));   /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;
        if (Temp < 0)
            Temp = 0;
        else if (Temp > 255)
            Temp = 255;

        output_ptr[out_offset] = (unsigned char)Temp;
    }
}


kernel void vp8_filter_block2d_bil_first_pass_kernel(
    __global unsigned char *src_base,
    int src_offset,
    __global int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
)
{
    uint tid = get_global_id(0);

    if (tid < output_width * output_height){
        global unsigned char *src_ptr = &src_base[src_offset];

        unsigned int i, j;
        __constant int *vp8_filter = &bilinear_filters[filter_offset*BILINEAR_ROW_SIZE];

        unsigned int out_row,out_offset;
        int src_increment = src_pixels_per_line - output_width;

        i = tid / output_width;
        j = tid % output_width;

        src_offset = i*(output_width+src_increment) + j;
        out_row = output_width * i;

        out_offset = out_row + j;

        /* Apply bilinear filter */
        output_ptr[out_offset] = (((int)src_ptr[src_offset]   * vp8_filter[0]) +
                 ((int)src_ptr[src_offset+1] * vp8_filter[1]) +
                 (VP8_FILTER_WEIGHT / 2)) >> VP8_FILTER_SHIFT;
    }
}

kernel void vp8_filter_block2d_bil_second_pass_kernel
(
    __global int *src_ptr,
    __global unsigned char *output_base,
    int output_offset,
    int output_pitch,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
)
{

    uint tid = get_global_id(0);

    if (tid < output_width * output_height){
        global unsigned char *output_ptr = &output_base[output_offset];

        unsigned int i, j;
        int Temp;
        __constant int *vp8_filter = &bilinear_filters[filter_offset*BILINEAR_ROW_SIZE];

        int out_offset;
        int src_offset;

        i = tid / output_width;
        j = tid % output_width;

        src_offset = i*(output_width) + j;
        out_offset = i*output_pitch + j;

        /* Apply filter */
        Temp = ((int)src_ptr[src_offset]         * vp8_filter[0]) +
               ((int)src_ptr[src_offset+output_width] * vp8_filter[1]) +
               (VP8_FILTER_WEIGHT / 2);

        output_ptr[out_offset++] = (unsigned int)(Temp >> VP8_FILTER_SHIFT);
    }
}




//Called from reconinter_cl.c
kernel void vp8_memcpy_kernel(
    global unsigned char *src_base,
    int src_offset,
    int src_stride,
    global unsigned char *dst_base,
    int dst_offset,
    int dst_stride,
    int num_bytes,
    int num_iter
){

    int i,r;
    global unsigned char *src = &src_base[src_offset];
    global unsigned char *dst = &dst_base[dst_offset];
    src_offset = dst_offset = 0;

    r = get_global_id(1);
    if (r < get_global_size(1)){
        i = get_global_id(0);
        if (i < get_global_size(0)){
            src_offset = r*src_stride + i;
            dst_offset = r*dst_stride + i;
            dst[dst_offset] = src[src_offset];
        }
    }
}

//Not used currently.
void vp8_memset_short(
    global short *mem,
    int offset,
    short newval,
    unsigned int size
)
{
    int tid = get_global_id(0);

    if (tid < (size/2)){
        mem[offset+tid/2] = newval;
    }
}



__kernel void vp8_bilinear_predict4x4_kernel
(
        __global unsigned char *src_base,
        int src_offset,
        int src_pixels_per_line,
        int xoffset,
        int yoffset,
        __global unsigned char *dst_base,
        int dst_offset,
        int dst_pitch,
        __global int *int_mem
)
{
    int Height = 4, Width = 4;

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_kernel(src_base, src_offset, int_mem, src_pixels_per_line, Height + 1, Width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_kernel(int_mem, dst_base, dst_offset, dst_pitch, Height, Width, yoffset);
}

__kernel void vp8_bilinear_predict8x8_kernel
(
    __global unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    __global unsigned char *dst_base,
    int dst_offset,
    int dst_pitch,
    __global int *int_mem
)
{
    int Height = 8, Width = 8;

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_kernel(src_base, src_offset, int_mem, src_pixels_per_line, Height + 1, Width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_kernel(int_mem, dst_base, dst_offset, dst_pitch, Height, Width, yoffset);

}

__kernel void vp8_bilinear_predict8x4_kernel
(
    __global unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    __global unsigned char *dst_base,
    int dst_offset,
    int dst_pitch,
    __global int *int_mem
)
{
    int Height = 4, Width = 8;

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_kernel(src_base, src_offset, int_mem, src_pixels_per_line, Height + 1, Width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_kernel(int_mem, dst_base, dst_offset, dst_pitch, Height, Width, yoffset);
}

__kernel void vp8_bilinear_predict16x16_kernel
(
    __global unsigned char *src_base,
    int src_offset,
    int src_pixels_per_line,
    int xoffset,
    int yoffset,
    __global unsigned char *dst_base,
    int dst_offset,
    int dst_pitch,
    __global int *int_mem
)
{

    int Height = 16, Width = 16;

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_bil_first_pass_kernel(src_base, src_offset, int_mem, src_pixels_per_line, Height + 1, Width, xoffset);

    /* then 1-D vertically... */
    vp8_filter_block2d_bil_second_pass_kernel(int_mem, dst_base, dst_offset, dst_pitch, Height, Width, yoffset);

}

void vp8_filter_block2d_first_pass(
    global unsigned char *src_base,
    int src_offset,
    local int *output_ptr,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
){
    uint tid = get_global_id(0);
    uint i = tid;

    int nthreads = get_global_size(0);
    int ngroups = get_num_groups(0);
    int lsize = get_local_size(0);

    global unsigned char *src_ptr = &src_base[src_offset];
    //Note that src_offset will be reset later, which is why we capture it now

    int Temp;

    if (tid < (output_width*output_height)){
        short8 filter = sub_pel_filter_values(filter_offset, sub_pel_filters);

        if (ngroups > 1){
            //This is generally only true on Apple CPU-CL, which gives a group
            //size of 1, regardless of the CPU core count.
            for (int iter=0; iter < ngroups; iter++){
                i = iter * lsize + get_local_id(0);
                src_offset = i + (i/output_width * (src_pixels_per_line - output_width));

#define USE_DOT_FILTERING 0
#if USE_DOT_FILTERING
            float8 src_data;
            float8 tmpfilter = convert_float8(filter);
            src_data.s0 = src_ptr[src_offset-2];
            src_data.s1 = src_ptr[src_offset-1];
            src_data.s2 = src_ptr[src_offset];
            src_data.s3 = src_ptr[src_offset+1];
            src_data.s4 = src_ptr[src_offset+2];
            src_data.s5 = src_ptr[src_offset+3];
            Temp = dot(src_data.s0123, tmpfilter.s0123) + dot(src_data.s01, tmpfilter.s45);
            Temp += (VP8_FILTER_WEIGHT >> 1);
#else
                Temp = (int)(src_ptr[src_offset - 2] * filter.s0) +
                       (int)(src_ptr[src_offset - 1] * filter.s1) +
                       (int)(src_ptr[src_offset]     * filter.s2) +
                       (int)(src_ptr[src_offset + 1] * filter.s3) +
                       (int)(src_ptr[src_offset + 2] * filter.s4) +
                       (int)(src_ptr[src_offset + 3] * filter.s5) +
                       (VP8_FILTER_WEIGHT >> 1);      /* Rounding */
#endif

                /* Normalize back to 0-255 */
                Temp >>= VP8_FILTER_SHIFT;

                Temp = clamp(Temp, 0, 255);

                output_ptr[i] = Temp;
            }
        } else {
            src_offset = i + (i/output_width * (src_pixels_per_line - output_width));

#if USE_DOT_FILTERING
            float8 src_data;
            float8 tmpfilter = convert_float8(filter);
            src_data.s0 = src_ptr[src_offset-2];
            src_data.s1 = src_ptr[src_offset-1];
            src_data.s2 = src_ptr[src_offset];
            src_data.s3 = src_ptr[src_offset+1];
            src_data.s4 = src_ptr[src_offset+2];
            src_data.s5 = src_ptr[src_offset+3];
            Temp = dot(src_data.s0123, tmpfilter.s0123) + dot(src_data.s01, tmpfilter.s45);
            Temp += (VP8_FILTER_WEIGHT >> 1);
#else
            Temp = (int)(src_ptr[src_offset - 2] * filter.s0) +
                   (int)(src_ptr[src_offset - 1] * filter.s1) +
                   (int)(src_ptr[src_offset]     * filter.s2) +
                   (int)(src_ptr[src_offset + 1] * filter.s3) +
                   (int)(src_ptr[src_offset + 2] * filter.s4) +
                   (int)(src_ptr[src_offset + 3] * filter.s5) +
                   (VP8_FILTER_WEIGHT >> 1);      /* Rounding */
#endif
            /* Normalize back to 0-255 */
            Temp >>= VP8_FILTER_SHIFT;

            Temp = clamp(Temp, 0, 255);

            output_ptr[i] = Temp;
        }
    }

    //Add a fence so that no 2nd pass stuff starts before 1st pass writes are done.
    barrier(CLK_LOCAL_MEM_FENCE);
    //write_mem_fence(CLK_LOCAL_MEM_FENCE);
}

void vp8_filter_block2d_second_pass
(
    local int *src_ptr,
    global unsigned char *output_base,
    int output_offset,
    int output_pitch,
    unsigned int src_pixels_per_line,
    unsigned int pixel_step,
    unsigned int output_height,
    unsigned int output_width,
    int filter_offset
) {

    global unsigned char *output_ptr = &output_base[output_offset];

    int out_offset; //Not same as output_offset...
    int src_offset;
    int Temp;
    int PS2 = 2*(int)pixel_step;
    int PS3 = 3*(int)pixel_step;

    unsigned int src_increment = src_pixels_per_line - output_width;

    uint i = get_global_id(0);

    short8 vp8_filter = sub_pel_filter_values(filter_offset, sub_pel_filters);
    
    if (i < (output_width * output_height)){
        out_offset = i/output_width;
        src_offset = out_offset;

        src_offset = i + (src_offset * src_increment);
        out_offset = i%output_width + (out_offset * output_pitch);

        /* Apply filter */
        Temp = ((int)src_ptr[src_offset - PS2] * vp8_filter.s0) +
           ((int)src_ptr[src_offset -(int)pixel_step] * vp8_filter.s1) +
           ((int)src_ptr[src_offset]                  * vp8_filter.s2) +
           ((int)src_ptr[src_offset + pixel_step]     * vp8_filter.s3) +
           ((int)src_ptr[src_offset + PS2]            * vp8_filter.s4) +
           ((int)src_ptr[src_offset + PS3]       * vp8_filter.s5) +
           (VP8_FILTER_WEIGHT >> 1);   /* Rounding */

        /* Normalize back to 0-255 */
        Temp = Temp >> VP8_FILTER_SHIFT;
        if (Temp < 0)
            Temp = 0;
        else if (Temp > 255)
            Temp = 255;

        output_ptr[out_offset] = (unsigned char)Temp;
    }
}

__kernel void vp8_sixtap_predict_kernel
(
    __global unsigned char  *src_ptr,
    int src_offset,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int dst_offset,
    int  dst_pitch
)
{

    local int FData[9*4];

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, src_offset, FData, src_pixels_per_line, 1, 9, 4, xoffset);

    /* then filter vertically... */
    vp8_filter_block2d_second_pass(&FData[8], dst_ptr, dst_offset, dst_pitch, 4, 4, 4, 4, yoffset);
}

__kernel void vp8_sixtap_predict8x8_kernel
(
    __global unsigned char  *src_ptr,
    int src_offset,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int dst_offset,
    int  dst_pitch
)
{
    local int FData[13*16];   /* Temp data bufffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, src_offset, FData, src_pixels_per_line, 1, 13, 8, xoffset);

    /* then filter vertically... */
    vp8_filter_block2d_second_pass(&FData[16], dst_ptr, dst_offset, dst_pitch, 8, 8, 8, 8, yoffset);

}

__kernel void vp8_sixtap_predict8x4_kernel
(
    __global unsigned char  *src_ptr,
    int src_offset,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int dst_offset,
    int  dst_pitch
)
{
    local int FData[13*16];   /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, src_offset, FData, src_pixels_per_line, 1, 9, 8, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[16], dst_ptr, dst_offset, dst_pitch, 8, 8, 4, 8, yoffset);
}

__kernel void vp8_sixtap_predict16x16_kernel
(
    __global unsigned char  *src_ptr,
    int src_offset,
    int  src_pixels_per_line,
    int  xoffset,
    int  yoffset,
    __global unsigned char *dst_ptr,
    int dst_offset,
    int  dst_pitch
)
{
    local int FData[21*24];   /* Temp data buffer used in filtering */

    /* First filter 1-D horizontally... */
    vp8_filter_block2d_first_pass(src_ptr, src_offset, FData, src_pixels_per_line, 1, 21, 16, xoffset);

    /* then filter verticaly... */
    vp8_filter_block2d_second_pass(&FData[32], dst_ptr, dst_offset, dst_pitch, 16, 16, 16, 16, yoffset);

    return;
}
