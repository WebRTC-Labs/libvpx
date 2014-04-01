/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

//for the decoder, all subpixel prediction is done in this file.
//
//Need to determine some sort of mechanism for easily determining SIXTAP/BILINEAR
//and what arguments to feed into the kernels. These kernels SHOULD be 2-pass,
//and ideally there'd be a data structure that determined what static arguments
//to pass in.
//
//Also, the only external functions being called here are the subpixel prediction
//functions. Hopefully this means no worrying about when to copy data back/forth.

#if 0

#include "vpx_config.h"
#if CONFIG_RUNTIME_CPU_DETECT
//#include "../onyxc_int.h"
#endif

#include "vp8_opencl.h"
#include "filter_cl.h"
#include "reconinter_cl.h"
#include "blockd_cl.h"

#include <stdio.h>

/* use this define on systems where unaligned int reads and writes are
 * not allowed, i.e. ARM architectures
 */
/*#define MUST_BE_ALIGNED*/

static const int bbb[4] = {0, 2, 8, 10};

#if 0  //Old memcpy code. Performs similarly to vp8_copy_mem_cl on a single block
static void vp8_memcpy(
    unsigned char *src_base,
    int src_offset,
    int src_stride,
    unsigned char *dst_base,
    int dst_offset,
    int dst_stride,
    int num_bytes,
    int num_iter
){

    int i,r;
    unsigned char *src = &src_base[src_offset];
    unsigned char *dst = &dst_base[dst_offset];
    src_offset = dst_offset = 0;

    for (r = 0; r < num_iter; r++){
        for (i = 0; i < num_bytes; i++){
            src_offset = r*src_stride + i;
            dst_offset = r*dst_stride + i;
            dst[dst_offset] = src[src_offset];
        }
    }
}
#endif

static void vp8_copy_mem_cl(
    cl_command_queue cq,
    cl_mem src_mem,
    int *src_offsets,
    int src_stride,
    cl_mem dst_mem,
    int *dst_offsets,
    int dst_stride,
    int num_bytes,
    int num_iter,
    int num_blocks
){

    int err,block;

#if MEM_COPY_KERNEL
    size_t global[3] = {num_bytes, num_iter, num_blocks};

    size_t local[3];
    local[0] = global[0];
    local[1] = global[1];
    local[2] = global[2];

    err  = clSetKernelArg(cl_data.vp8_memcpy_kernel, 0, sizeof (cl_mem), &src_mem);
    err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 2, sizeof (int), &src_stride);
    err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 3, sizeof (cl_mem), &dst_mem);
    err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 5, sizeof (int), &dst_stride);
    err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 6, sizeof (int), &num_bytes);
    err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 7, sizeof (int), &num_iter);
    VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
        "Error: Failed to set kernel arguments!\n",
        return,
    );

    for (block = 0; block < num_blocks; block++){

        /* Set kernel arguments */
        err = clSetKernelArg(cl_data.vp8_memcpy_kernel, 1, sizeof (int), &src_offsets[block]);
        err |= clSetKernelArg(cl_data.vp8_memcpy_kernel, 4, sizeof (int), &dst_offsets[block]);
        VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
            "Error: Failed to set kernel arguments!\n",
            return,
        );

        /* Execute the kernel */
        if (num_bytes * num_iter > cl_data.vp8_memcpy_kernel_size){
            err = clEnqueueNDRangeKernel( cq, cl_data.vp8_memcpy_kernel, 2, NULL, global, NULL , 0, NULL, NULL);
        } else {
            err = clEnqueueNDRangeKernel( cq, cl_data.vp8_memcpy_kernel, 2, NULL, global, local , 0, NULL, NULL);
        }

        VP8_CL_CHECK_SUCCESS( cq, err != CL_SUCCESS,
            "Error: Failed to execute kernel!\n",
            return,
        );
    }
#else
    int iter;

    for (block=0; block < num_blocks; block++){
        const size_t src_origin[3] = { src_offsets[block] % src_stride, src_offsets[block] / src_stride, 0};
        const size_t dst_origin[3] = { dst_offsets[block] % dst_stride, dst_offsets[block] / dst_stride, 0};
        const size_t region[3] = {num_bytes, num_iter, 1};

        err = clEnqueueCopyBufferRect( cq, src_mem, dst_mem,
                    src_origin, dst_origin, region,
                    src_stride,
                    src_stride*(src_origin[1]+iter),
                    dst_stride,
                    dst_stride*(dst_origin[1]+iter),
                    0,
                    NULL,
                    NULL);

        VP8_CL_CHECK_SUCCESS(cq, err != CL_SUCCESS, "Error copying between buffers\n",
                ,
        );
    }
#endif
}

static void vp8_build_inter_predictors_b_cl(MACROBLOCKD *x, BLOCKD *d, int pitch)
{
    unsigned char *ptr_base = *(d->base_pre);
    int ptr_offset = d->pre + (d->bmi.mv.as_mv.row >> 3) * d->pre_stride + (d->bmi.mv.as_mv.col >> 3);

    vp8_subpix_cl_fn_t sppf;

    int pre_dist = *d->base_pre - x->pre.buffer_alloc;
    cl_mem pre_mem = x->pre.buffer_mem;
    int pre_off = pre_dist+ptr_offset;

    if (d->sixtap_filter == CL_TRUE)
        sppf = vp8_sixtap_predict4x4_cl;
    else
        sppf = vp8_bilinear_predict4x4_cl;

    //ptr_base a.k.a. d->base_pre is the start of the
    //Macroblock's y_buffer, u_buffer, or v_buffer

    if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7)
    {
        sppf(d->cl_commands, ptr_base, pre_mem, pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor_base, d->cl_predictor_mem, d->predictor_offset, pitch);
    }
    else
    {
        vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, d->pre_stride,d->cl_predictor_mem, &d->predictor_offset,pitch,4,4,1);
    }
}


static void vp8_build_inter_predictors4b_cl(MACROBLOCKD *x, BLOCKD *d, int pitch)
{
    unsigned char *ptr_base = *(d->base_pre);
    int ptr_offset = d->pre + (d->bmi.mv.as_mv.row >> 3) * d->pre_stride + (d->bmi.mv.as_mv.col >> 3);

    int pre_dist = *d->base_pre - x->pre.buffer_alloc;
    cl_mem pre_mem = x->pre.buffer_mem;
    int pre_off = pre_dist + ptr_offset;

    //If there's motion in the bottom 8 subpixels, need to do subpixel prediction
    if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7)
    {
            if (d->sixtap_filter == CL_TRUE)
                vp8_sixtap_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor_base, d->cl_predictor_mem, d->predictor_offset, pitch);
            else
                vp8_bilinear_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor_base, d->cl_predictor_mem, d->predictor_offset, pitch);
    }
    //Otherwise copy memory directly from src to dest
    else
    {
        vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, d->pre_stride, d->cl_predictor_mem, &d->predictor_offset, pitch, 8, 8, 1);
    }


}

static void vp8_build_inter_predictors2b_cl(MACROBLOCKD *x, BLOCKD *d, int pitch)
{
    unsigned char *ptr_base = *(d->base_pre);

    int ptr_offset = d->pre + (d->bmi.mv.as_mv.row >> 3) * d->pre_stride + (d->bmi.mv.as_mv.col >> 3);

    int pre_dist = *d->base_pre - x->pre.buffer_alloc;
    cl_mem pre_mem = x->pre.buffer_mem;
    int pre_off = pre_dist+ptr_offset;

    if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7)
    {
        if (d->sixtap_filter == CL_TRUE)
            vp8_sixtap_predict8x4_cl(d->cl_commands,ptr_base,pre_mem,pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor_base, d->cl_predictor_mem, d->predictor_offset, pitch);
        else
            vp8_bilinear_predict8x4_cl(d->cl_commands,ptr_base,pre_mem,pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor_base, d->cl_predictor_mem, d->predictor_offset, pitch);
    }
    else
    {
        vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, d->pre_stride, d->cl_predictor_mem, &d->predictor_offset, pitch, 8, 4, 1);
    }
}


void vp8_build_inter_predictors_mbuv_cl(MACROBLOCKD *x)
{
    int i;

    vp8_cl_mb_prep(x, PREDICTOR|PRE_BUF);

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->cl_commands);
#endif

    if (x->mode_info_context->mbmi.ref_frame != INTRA_FRAME &&
        x->mode_info_context->mbmi.mode != SPLITMV)
    {

        unsigned char *pred_base = x->predictor;
        int upred_offset = 256;
        int vpred_offset = 320;

        int mv_row = x->block[16].bmi.mv.as_mv.row;
        int mv_col = x->block[16].bmi.mv.as_mv.col;
        int offset;

        unsigned char *pre_base = x->pre.buffer_alloc;
        cl_mem pre_mem = x->pre.buffer_mem;
        int upre_off = x->pre.u_buffer - pre_base;
        int vpre_off = x->pre.v_buffer - pre_base;
        int pre_stride = x->block[16].pre_stride;

        offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);

        if ((mv_row | mv_col) & 7)
        {
            if (cl_initialized == CL_SUCCESS && x->sixtap_filter == CL_TRUE){
                vp8_sixtap_predict8x8_cl(x->block[16].cl_commands,pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, upred_offset, 8);
                vp8_sixtap_predict8x8_cl(x->block[20].cl_commands,pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, vpred_offset, 8);
            }
            else{
                vp8_bilinear_predict8x8_cl(x->block[16].cl_commands,pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, upred_offset, 8);
                vp8_bilinear_predict8x8_cl(x->block[20].cl_commands,pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, vpred_offset, 8);
            }
        }
        else
        {
            int pre_offsets[2] = {upre_off+offset, vpre_off+offset};
            int pred_offsets[2] = {upred_offset,vpred_offset};
            vp8_copy_mem_cl(x->block[16].cl_commands, pre_mem, pre_offsets, pre_stride, x->cl_predictor_mem, pred_offsets, 8, 8, 8, 2);
        }
    }
    else
    {
        // Can probably batch these operations as well, but not tested in decoder
        // (or at least the test videos I've been using.
        for (i = 16; i < 24; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];
            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                vp8_build_inter_predictors2b_cl(x, d0, 8);
            else
            {
                vp8_build_inter_predictors_b_cl(x, d0, 8);
                vp8_build_inter_predictors_b_cl(x, d1, 8);
            }
        }
    }

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->block[0].cl_commands);
    VP8_CL_FINISH(x->block[16].cl_commands);
    VP8_CL_FINISH(x->block[20].cl_commands);
#endif

    vp8_cl_mb_finish(x, PREDICTOR);
}

void vp8_build_inter_predictors_mb_cl(MACROBLOCKD *x)
{
    //If CL is running in encoder, need to call following before proceeding.
    //vp8_cl_mb_prep(x, PRE_BUF);

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->cl_commands);
#endif

    if (x->mode_info_context->mbmi.ref_frame != INTRA_FRAME &&
        x->mode_info_context->mbmi.mode != SPLITMV)
    {
        int offset;
        unsigned char *pred_base = x->predictor;
        int upred_offset = 256;
        int vpred_offset = 320;

        int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
        int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
        int pre_stride = x->block[0].pre_stride;

        unsigned char *pre_base = x->pre.buffer_alloc;
        cl_mem pre_mem = x->pre.buffer_mem;
        int ypre_off = x->pre.y_buffer - pre_base + (mv_row >> 3) * pre_stride + (mv_col >> 3);
        int upre_off = x->pre.u_buffer - pre_base;
        int vpre_off = x->pre.v_buffer - pre_base;

        if ((mv_row | mv_col) & 7)
        {
            if (cl_initialized == CL_SUCCESS && x->sixtap_filter == CL_TRUE){
                vp8_sixtap_predict16x16_cl(x->block[0].cl_commands, pre_base, pre_mem, ypre_off, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, 0, 16);
            }
            else
                vp8_bilinear_predict16x16_cl(x->block[0].cl_commands, pre_base, pre_mem,  ypre_off, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, 0, 16);
        }
        else
        {
            //16x16 copy
            int pred_off = 0;
            vp8_copy_mem_cl(x->block[0].cl_commands, pre_mem, &ypre_off, pre_stride, x->cl_predictor_mem, &pred_off, 16, 16, 16, 1);
        }


        mv_row = x->block[16].bmi.mv.as_mv.row;
        mv_col = x->block[16].bmi.mv.as_mv.col;
        pre_stride >>= 1;
        offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);

        if ((mv_row | mv_col) & 7)
        {
            if (x->sixtap_filter == CL_TRUE){
                vp8_sixtap_predict8x8_cl(x->block[16].cl_commands, pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, upred_offset, 8);
                vp8_sixtap_predict8x8_cl(x->block[20].cl_commands, pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, vpred_offset, 8);
            }
            else {
                vp8_bilinear_predict8x8_cl(x->block[16].cl_commands, pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, upred_offset, 8);
                vp8_bilinear_predict8x8_cl(x->block[20].cl_commands, pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, pred_base, x->cl_predictor_mem, vpred_offset, 8);
            }
        }
        else
        {
            int pre_off = upre_off + offset;
            vp8_copy_mem_cl(x->block[16].cl_commands, pre_mem, &pre_off, pre_stride, x->cl_predictor_mem, &upred_offset, 8, 8, 8, 1);
            pre_off = vpre_off + offset;
            vp8_copy_mem_cl(x->block[20].cl_commands, pre_mem, &pre_off, pre_stride, x->cl_predictor_mem, &vpred_offset, 8, 8, 8, 1);
        }
    }
    else
    {
        int i;

        if (x->mode_info_context->mbmi.partitioning < 3)
        {
            for (i = 0; i < 4; i++)
            {
                BLOCKD *d = &x->block[bbb[i]];
                vp8_build_inter_predictors4b_cl(x, d, 16);
            }
        }
        else
        {
            /* This loop can be done in any order... No dependencies.*/
            /* Also, d0/d1 can be decoded simultaneously */
            for (i = 0; i < 16; i += 2)
            {
                BLOCKD *d0 = &x->block[i];
                BLOCKD *d1 = &x->block[i+1];

                if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                    vp8_build_inter_predictors2b_cl(x, d0, 16);
                else
                {
                    vp8_build_inter_predictors_b_cl(x, d0, 16);
                    vp8_build_inter_predictors_b_cl(x, d1, 16);
                }
            }
        }

        /* Another case of re-orderable/batchable loop */
        for (i = 16; i < 24; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];

            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                vp8_build_inter_predictors2b_cl(x, d0, 8);
            else
            {
                vp8_build_inter_predictors_b_cl(x, d0, 8);
                vp8_build_inter_predictors_b_cl(x, d1, 8);
            }
        }
    }

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->block[0].cl_commands);
    VP8_CL_FINISH(x->block[16].cl_commands);
    VP8_CL_FINISH(x->block[20].cl_commands);
#endif

    vp8_cl_mb_finish(x, PREDICTOR);
}


/* The following functions are written for skip_recon_mb() to call. Since there is no recon in this
 * situation, we can write the result directly to dst buffer instead of writing it to predictor
 * buffer and then copying it to dst buffer.
 */
static void vp8_build_inter_predictors_b_s_cl(MACROBLOCKD *x, BLOCKD *d, int dst_offset)
{
    unsigned char *ptr_base = *(d->base_pre);
    int dst_stride = d->dst_stride;
    int pre_stride = d->pre_stride;
    int ptr_offset = d->pre + (d->bmi.mv.as_mv.row >> 3) * d->pre_stride + (d->bmi.mv.as_mv.col >> 3);
    vp8_subpix_cl_fn_t sppf;

    int pre_dist = *d->base_pre - x->pre.buffer_alloc;
    cl_mem pre_mem = x->pre.buffer_mem;
    cl_mem dst_mem = x->dst.buffer_mem;

    if (d->sixtap_filter == CL_TRUE){
        sppf = vp8_sixtap_predict4x4_cl;
    } else
        sppf = vp8_bilinear_predict4x4_cl;

    if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7)
    {
        sppf(d->cl_commands, ptr_base, pre_mem, pre_dist+ptr_offset, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, NULL, dst_mem, dst_offset, dst_stride);
    }
    else
    {
        int pre_off = pre_dist+ptr_offset;
        vp8_copy_mem_cl(d->cl_commands, pre_mem,&pre_off,pre_stride, dst_mem, &dst_offset,dst_stride,4,4,1);
    }
}


void vp8_build_inter_predictors_mb_s_cl(MACROBLOCKD *x)
{
    cl_mem dst_mem = NULL;
    cl_mem pre_mem = x->pre.buffer_mem;

    unsigned char *dst_base = x->dst.buffer_alloc;
    int ydst_off = x->dst.y_buffer - dst_base;
    int udst_off = x->dst.u_buffer - dst_base;
    int vdst_off = x->dst.v_buffer - dst_base;

    dst_mem = x->dst.buffer_mem;
    vp8_cl_mb_prep(x, DST_BUF);

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->cl_commands);
#endif

    if (x->mode_info_context->mbmi.mode != SPLITMV)
    {
        int offset;
        unsigned char *pre_base = x->pre.buffer_alloc;
        int ypre_off = x->pre.y_buffer - pre_base;
        int upre_off = x->pre.u_buffer - pre_base;
        int vpre_off = x->pre.v_buffer - pre_base;

        int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
        int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
        int pre_stride = x->dst.y_stride;

        int ptr_offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);

        if ((mv_row | mv_col) & 7)
        {
            if (x->sixtap_filter == CL_TRUE){
                vp8_sixtap_predict16x16_cl(x->block[0].cl_commands, pre_base, pre_mem, ypre_off+ptr_offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
            }
            else
                vp8_bilinear_predict16x16_cl(x->block[0].cl_commands, pre_base, pre_mem, ypre_off+ptr_offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
        }
        else
        {
            int pre_off = ypre_off+ptr_offset;
            vp8_copy_mem_cl(x->block[0].cl_commands, pre_mem, &pre_off, pre_stride, dst_mem, &ydst_off, x->dst.y_stride, 16, 16, 1);
        }

        mv_row = x->block[16].bmi.mv.as_mv.row;
        mv_col = x->block[16].bmi.mv.as_mv.col;
        pre_stride >>= 1;
        offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);

        if ((mv_row | mv_col) & 7)
        {
            if (x->sixtap_filter == CL_TRUE){
                vp8_sixtap_predict8x8_cl(x->block[16].cl_commands, pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, udst_off, x->dst.uv_stride);
                vp8_sixtap_predict8x8_cl(x->block[20].cl_commands, pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, vdst_off, x->dst.uv_stride);
            } else {
                vp8_bilinear_predict8x8_cl(x->block[16].cl_commands, pre_base, pre_mem, upre_off+offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, udst_off, x->dst.uv_stride);
                vp8_bilinear_predict8x8_cl(x->block[20].cl_commands, pre_base, pre_mem, vpre_off+offset, pre_stride, mv_col & 7, mv_row & 7, dst_base, dst_mem, vdst_off, x->dst.uv_stride);
            }
        }
        else
        {
            int pre_offsets[2] = {upre_off+offset, vpre_off+offset};
            int dst_offsets[2] = {udst_off,vdst_off};
            vp8_copy_mem_cl(x->block[16].cl_commands, pre_mem, pre_offsets, pre_stride, dst_mem, dst_offsets, x->dst.uv_stride, 8, 8, 2);
        }

    }
    else
    {
        /* note: this whole ELSE part is not executed at all. So, no way to test the correctness of my modification. Later,
         * if sth is wrong, go back to what it is in build_inter_predictors_mb.
         *
         * ACW: Not sure who the above comment belongs to, but it is
         *      accurate for the decoder. Verified by reverse trace of source
         */
        int i;

        if (x->mode_info_context->mbmi.partitioning < 3)
        {
            for (i = 0; i < 4; i++)
            {
                BLOCKD *d = &x->block[bbb[i]];

                {
                    unsigned char *ptr_base = *(d->base_pre);
                    int pre_off = ptr_base - x->pre.buffer_alloc;

                    int ptr_offset = d->pre + (d->bmi.mv.as_mv.row >> 3) * d->pre_stride + (d->bmi.mv.as_mv.col >> 3);

                    pre_off += ptr_offset;

                    if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7)
                    {
                        if (x->sixtap_filter == CL_TRUE)
                            vp8_sixtap_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
                        else
                            vp8_bilinear_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off, d->pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
                    }
                    else
                    {
                        vp8_copy_mem_cl(x->block[0].cl_commands, pre_mem, &pre_off, d->pre_stride, dst_mem, &ydst_off, x->dst.y_stride, 8, 8, 1);
                    }
                }
            }
        }
        else
        {
            for (i = 0; i < 16; i += 2)
            {
                BLOCKD *d0 = &x->block[i];
                BLOCKD *d1 = &x->block[i+1];

                if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                {
                    /*vp8_build_inter_predictors2b(x, d0, 16);*/
                    unsigned char *ptr_base = *(d0->base_pre);

                    int pre_off = ptr_base - x->pre.buffer_alloc;

                    int ptr_offset = d0->pre + (d0->bmi.mv.as_mv.row >> 3) * d0->pre_stride + (d0->bmi.mv.as_mv.col >> 3);
                    pre_off += ptr_offset;

                    if ( (d0->bmi.mv.as_mv.row | d0->bmi.mv.as_mv.col) & 7)
                    {
                        if (d0->sixtap_filter == CL_TRUE)
                            vp8_sixtap_predict8x4_cl(d0->cl_commands, ptr_base, pre_mem, pre_off, d0->pre_stride, d0->bmi.mv.as_mv.col & 7, d0->bmi.mv.as_mv.row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
                        else
                            vp8_bilinear_predict8x4_cl(d0->cl_commands, ptr_base, pre_mem,pre_off, d0->pre_stride, d0->bmi.mv.as_mv.col & 7, d0->bmi.mv.as_mv.row & 7, dst_base, dst_mem, ydst_off, x->dst.y_stride);
                    }
                    else
                    {
                        vp8_copy_mem_cl(x->block[0].cl_commands, pre_mem, &pre_off, d0->pre_stride, dst_mem, &ydst_off, x->dst.y_stride, 8, 4, 1);
                    }
                }
                else
                {
                    vp8_build_inter_predictors_b_s_cl(x,d0, ydst_off);
                    vp8_build_inter_predictors_b_s_cl(x,d1, ydst_off);
                }
            }
        }

        for (i = 16; i < 24; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];

            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            {
                /*vp8_build_inter_predictors2b(x, d0, 8);*/
                unsigned char *ptr_base = *(d0->base_pre);
                int ptr_offset = d0->pre + (d0->bmi.mv.as_mv.row >> 3) * d0->pre_stride + (d0->bmi.mv.as_mv.col >> 3);
                int pre_off = ptr_base - x->pre.buffer_alloc + ptr_offset;

                if ( (d0->bmi.mv.as_mv.row | d0->bmi.mv.as_mv.col) & 7)
                {
                    if (d0->sixtap_filter || CL_TRUE)
                        vp8_sixtap_predict8x4_cl(d0->cl_commands, ptr_base, pre_mem, pre_off, d0->pre_stride,
                            d0->bmi.mv.as_mv.col & 7, d0->bmi.mv.as_mv.row & 7,
                            dst_base, dst_mem, ydst_off, x->dst.uv_stride);
                    else
                        vp8_bilinear_predict8x4_cl(d0->cl_commands, ptr_base, pre_mem, pre_off, d0->pre_stride,
                            d0->bmi.mv.as_mv.col & 7, d0->bmi.mv.as_mv.row & 7,
                            dst_base, dst_mem, ydst_off, x->dst.uv_stride);
                }
                else
                {
                    vp8_copy_mem_cl(x->block[0].cl_commands, pre_mem, &pre_off,
                        d0->pre_stride, dst_mem, &ydst_off, x->dst.uv_stride, 8, 4, 1);
                }
            }
            else
            {
                vp8_build_inter_predictors_b_s_cl(x,d0, ydst_off);
                vp8_build_inter_predictors_b_s_cl(x,d1, ydst_off);
            }
        } //end for
    }

#if !ONE_CQ_PER_MB
    VP8_CL_FINISH(x->block[0].cl_commands);
    VP8_CL_FINISH(x->block[16].cl_commands);
    VP8_CL_FINISH(x->block[20].cl_commands);
#endif

    vp8_cl_mb_finish(x, DST_BUF);
}
#endif

/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include <limits.h>
#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx/vpx_integer.h"
#include "blockd_cl.h"
#include "reconinter_cl.h"
#if CONFIG_RUNTIME_CPU_DETECT
#include "../onyxc_int.h"
#endif

void vp8_copy_mem16x16_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{

    int r;

    for (r = 0; r < 16; r++)
    {
#if !(CONFIG_FAST_UNALIGNED)
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
        dst[5] = src[5];
        dst[6] = src[6];
        dst[7] = src[7];
        dst[8] = src[8];
        dst[9] = src[9];
        dst[10] = src[10];
        dst[11] = src[11];
        dst[12] = src[12];
        dst[13] = src[13];
        dst[14] = src[14];
        dst[15] = src[15];

#else
        ((uint32_t *)dst)[0] = ((uint32_t *)src)[0] ;
        ((uint32_t *)dst)[1] = ((uint32_t *)src)[1] ;
        ((uint32_t *)dst)[2] = ((uint32_t *)src)[2] ;
        ((uint32_t *)dst)[3] = ((uint32_t *)src)[3] ;

#endif
        src += src_stride;
        dst += dst_stride;

    }

}

void vp8_copy_mem8x8_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{
    int r;

    for (r = 0; r < 8; r++)
    {
#if !(CONFIG_FAST_UNALIGNED)
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
        dst[5] = src[5];
        dst[6] = src[6];
        dst[7] = src[7];
#else
        ((uint32_t *)dst)[0] = ((uint32_t *)src)[0] ;
        ((uint32_t *)dst)[1] = ((uint32_t *)src)[1] ;
#endif
        src += src_stride;
        dst += dst_stride;

    }

}

void vp8_copy_mem8x4_c(
    unsigned char *src,
    int src_stride,
    unsigned char *dst,
    int dst_stride)
{
    int r;

    for (r = 0; r < 4; r++)
    {
#if !(CONFIG_FAST_UNALIGNED)
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
        dst[4] = src[4];
        dst[5] = src[5];
        dst[6] = src[6];
        dst[7] = src[7];
#else
        ((uint32_t *)dst)[0] = ((uint32_t *)src)[0] ;
        ((uint32_t *)dst)[1] = ((uint32_t *)src)[1] ;
#endif
        src += src_stride;
        dst += dst_stride;

    }

}


void vp8_build_inter_predictors_b(BLOCKD *d, int pitch, unsigned char *base_pre, int pre_stride, vp8_subpix_fn_t sppf)
{
    int r;
    unsigned char *pred_ptr = d->predictor;
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        sppf(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, pred_ptr, pitch);
    }
    else
    {
        for (r = 0; r < 4; r++)
        {
            pred_ptr[0]  = ptr[0];
            pred_ptr[1]  = ptr[1];
            pred_ptr[2]  = ptr[2];
            pred_ptr[3]  = ptr[3];
            pred_ptr     += pitch;
            ptr         += pre_stride;
        }
    }
}

static void build_inter_predictors4b(MACROBLOCKD *x, BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride)
{
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        x->subpixel_predict8x8(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        vp8_copy_mem8x8(ptr, pre_stride, dst, dst_stride);
    }
}

static void build_inter_predictors2b(MACROBLOCKD *x, BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride)
{
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        x->subpixel_predict8x4(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        vp8_copy_mem8x4(ptr, pre_stride, dst, dst_stride);
    }
}

static void build_inter_predictors_b(BLOCKD *d, unsigned char *dst, int dst_stride, unsigned char *base_pre, int pre_stride, vp8_subpix_fn_t sppf)
{
    int r;
    unsigned char *ptr;
    ptr = base_pre + d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

    if (d->bmi.mv.as_mv.row & 7 || d->bmi.mv.as_mv.col & 7)
    {
        sppf(ptr, pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, dst, dst_stride);
    }
    else
    {
        for (r = 0; r < 4; r++)
        {
          dst[0]  = ptr[0];
          dst[1]  = ptr[1];
          dst[2]  = ptr[2];
          dst[3]  = ptr[3];
          dst     += dst_stride;
          ptr     += pre_stride;
        }
    }
}


/*encoder only*/
void vp8_build_inter16x16_predictors_mbuv(MACROBLOCKD *x)
{
    unsigned char *uptr, *vptr;
    unsigned char *upred_ptr = &x->predictor[256];
    unsigned char *vpred_ptr = &x->predictor[320];

    int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
    int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
    int offset;
    int pre_stride = x->pre.uv_stride;

    /* calc uv motion vectors */
    mv_row += 1 | (mv_row >> (sizeof(int) * CHAR_BIT - 1));
    mv_col += 1 | (mv_col >> (sizeof(int) * CHAR_BIT - 1));
    mv_row /= 2;
    mv_col /= 2;
    mv_row &= x->fullpixel_mask;
    mv_col &= x->fullpixel_mask;

    offset = (mv_row >> 3) * pre_stride + (mv_col >> 3);
    uptr = x->pre.u_buffer + offset;
    vptr = x->pre.v_buffer + offset;

    if ((mv_row | mv_col) & 7)
    {
        x->subpixel_predict8x8(uptr, pre_stride, mv_col & 7, mv_row & 7, upred_ptr, 8);
        x->subpixel_predict8x8(vptr, pre_stride, mv_col & 7, mv_row & 7, vpred_ptr, 8);
    }
    else
    {
        vp8_copy_mem8x8(uptr, pre_stride, upred_ptr, 8);
        vp8_copy_mem8x8(vptr, pre_stride, vpred_ptr, 8);
    }
}

/*encoder only*/
void vp8_build_inter4x4_predictors_mbuv(MACROBLOCKD *x)
{
    int i, j;
    int pre_stride = x->pre.uv_stride;
    unsigned char *base_pre;

    /* build uv mvs */
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            int yoffset = i * 8 + j * 2;
            int uoffset = 16 + i * 2 + j;
            int voffset = 20 + i * 2 + j;

            int temp;

            temp = x->block[yoffset  ].bmi.mv.as_mv.row
                   + x->block[yoffset+1].bmi.mv.as_mv.row
                   + x->block[yoffset+4].bmi.mv.as_mv.row
                   + x->block[yoffset+5].bmi.mv.as_mv.row;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.row = (temp / 8) & x->fullpixel_mask;

            temp = x->block[yoffset  ].bmi.mv.as_mv.col
                   + x->block[yoffset+1].bmi.mv.as_mv.col
                   + x->block[yoffset+4].bmi.mv.as_mv.col
                   + x->block[yoffset+5].bmi.mv.as_mv.col;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.col = (temp / 8) & x->fullpixel_mask;

            x->block[voffset].bmi.mv.as_int = x->block[uoffset].bmi.mv.as_int;
        }
    }

    base_pre = x->pre.u_buffer;
    for (i = 16; i < 20; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, d0->predictor, 8, base_pre, pre_stride);
        else
        {
            vp8_build_inter_predictors_b(d0, 8, base_pre, pre_stride, x->subpixel_predict);
            vp8_build_inter_predictors_b(d1, 8, base_pre, pre_stride, x->subpixel_predict);
        }
    }

    base_pre = x->pre.v_buffer;
    for (i = 20; i < 24; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, d0->predictor, 8, base_pre, pre_stride);
        else
        {
            vp8_build_inter_predictors_b(d0, 8, base_pre, pre_stride, x->subpixel_predict);
            vp8_build_inter_predictors_b(d1, 8, base_pre, pre_stride, x->subpixel_predict);
        }
    }
}


/*encoder only*/
void vp8_build_inter16x16_predictors_mby(MACROBLOCKD *x,
                                         unsigned char *dst_y,
                                         int dst_ystride)
{
    unsigned char *ptr_base;
    unsigned char *ptr;
    int mv_row = x->mode_info_context->mbmi.mv.as_mv.row;
    int mv_col = x->mode_info_context->mbmi.mv.as_mv.col;
    int pre_stride = x->pre.y_stride;

    ptr_base = x->pre.y_buffer;
    ptr = ptr_base + (mv_row >> 3) * pre_stride + (mv_col >> 3);

    if ((mv_row | mv_col) & 7)
    {
        x->subpixel_predict16x16(ptr, pre_stride, mv_col & 7, mv_row & 7,
                                 dst_y, dst_ystride);
    }
    else
    {
        vp8_copy_mem16x16(ptr, pre_stride, dst_y,
            dst_ystride);
    }
}

static void clamp_mv_to_umv_border(MV *mv, const MACROBLOCKD *xd)
{
    /* If the MV points so far into the UMV border that no visible pixels
     * are used for reconstruction, the subpel part of the MV can be
     * discarded and the MV limited to 16 pixels with equivalent results.
     *
     * This limit kicks in at 19 pixels for the top and left edges, for
     * the 16 pixels plus 3 taps right of the central pixel when subpel
     * filtering. The bottom and right edges use 16 pixels plus 2 pixels
     * left of the central pixel when filtering.
     */
    if (mv->col < (xd->mb_to_left_edge - (19 << 3)))
        mv->col = xd->mb_to_left_edge - (16 << 3);
    else if (mv->col > xd->mb_to_right_edge + (18 << 3))
        mv->col = xd->mb_to_right_edge + (16 << 3);

    if (mv->row < (xd->mb_to_top_edge - (19 << 3)))
        mv->row = xd->mb_to_top_edge - (16 << 3);
    else if (mv->row > xd->mb_to_bottom_edge + (18 << 3))
        mv->row = xd->mb_to_bottom_edge + (16 << 3);
}

/* A version of the above function for chroma block MVs.*/
static void clamp_uvmv_to_umv_border(MV *mv, const MACROBLOCKD *xd)
{
    mv->col = (2*mv->col < (xd->mb_to_left_edge - (19 << 3))) ?
        (xd->mb_to_left_edge - (16 << 3)) >> 1 : mv->col;
    mv->col = (2*mv->col > xd->mb_to_right_edge + (18 << 3)) ?
        (xd->mb_to_right_edge + (16 << 3)) >> 1 : mv->col;

    mv->row = (2*mv->row < (xd->mb_to_top_edge - (19 << 3))) ?
        (xd->mb_to_top_edge - (16 << 3)) >> 1 : mv->row;
    mv->row = (2*mv->row > xd->mb_to_bottom_edge + (18 << 3)) ?
        (xd->mb_to_bottom_edge + (16 << 3)) >> 1 : mv->row;
}

void vp8_build_inter16x16_predictors_mb(MACROBLOCKD *x,
                                        unsigned char *dst_y,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v,
                                        int dst_ystride,
                                        int dst_uvstride)
{
    int offset;
    unsigned char *ptr;
    unsigned char *uptr, *vptr;

    int_mv _16x16mv;

    unsigned char *ptr_base = x->pre.y_buffer;
    int pre_stride = x->pre.y_stride;

    _16x16mv.as_int = x->mode_info_context->mbmi.mv.as_int;

    if (x->mode_info_context->mbmi.need_to_clamp_mvs)
    {
        clamp_mv_to_umv_border(&_16x16mv.as_mv, x);
    }

    ptr = ptr_base + ( _16x16mv.as_mv.row >> 3) * pre_stride + (_16x16mv.as_mv.col >> 3);

    if ( _16x16mv.as_int & 0x00070007)
    {
        x->subpixel_predict16x16(ptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_y, dst_ystride);
    }
    else
    {
        vp8_copy_mem16x16(ptr, pre_stride, dst_y, dst_ystride);
    }

    /* calc uv motion vectors */
    _16x16mv.as_mv.row += 1 | (_16x16mv.as_mv.row >> (sizeof(int) * CHAR_BIT - 1));
    _16x16mv.as_mv.col += 1 | (_16x16mv.as_mv.col >> (sizeof(int) * CHAR_BIT - 1));
    _16x16mv.as_mv.row /= 2;
    _16x16mv.as_mv.col /= 2;
    _16x16mv.as_mv.row &= x->fullpixel_mask;
    _16x16mv.as_mv.col &= x->fullpixel_mask;

    pre_stride >>= 1;
    offset = ( _16x16mv.as_mv.row >> 3) * pre_stride + (_16x16mv.as_mv.col >> 3);
    uptr = x->pre.u_buffer + offset;
    vptr = x->pre.v_buffer + offset;

    if ( _16x16mv.as_int & 0x00070007)
    {
        x->subpixel_predict8x8(uptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_u, dst_uvstride);
        x->subpixel_predict8x8(vptr, pre_stride, _16x16mv.as_mv.col & 7,  _16x16mv.as_mv.row & 7, dst_v, dst_uvstride);
    }
    else
    {
        vp8_copy_mem8x8(uptr, pre_stride, dst_u, dst_uvstride);
        vp8_copy_mem8x8(vptr, pre_stride, dst_v, dst_uvstride);
    }
}

static void build_inter4x4_predictors_mb(MACROBLOCKD *x)
{
    int i;
    unsigned char *base_dst = x->dst.y_buffer;
    unsigned char *base_pre = x->pre.y_buffer;

    if (x->mode_info_context->mbmi.partitioning < 3)
    {
        BLOCKD *b;
        int dst_stride = x->dst.y_stride;

        x->block[ 0].bmi = x->mode_info_context->bmi[ 0];
        x->block[ 2].bmi = x->mode_info_context->bmi[ 2];
        x->block[ 8].bmi = x->mode_info_context->bmi[ 8];
        x->block[10].bmi = x->mode_info_context->bmi[10];
        if (x->mode_info_context->mbmi.need_to_clamp_mvs)
        {
            clamp_mv_to_umv_border(&x->block[ 0].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[ 2].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[ 8].bmi.mv.as_mv, x);
            clamp_mv_to_umv_border(&x->block[10].bmi.mv.as_mv, x);
        }

        b = &x->block[ 0];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[ 2];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[ 8];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
        b = &x->block[10];
        build_inter_predictors4b(x, b, base_dst + b->offset, dst_stride, base_pre, dst_stride);
    }
    else
    {
        for (i = 0; i < 16; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];
            int dst_stride = x->dst.y_stride;

            x->block[i+0].bmi = x->mode_info_context->bmi[i+0];
            x->block[i+1].bmi = x->mode_info_context->bmi[i+1];
            if (x->mode_info_context->mbmi.need_to_clamp_mvs)
            {
                clamp_mv_to_umv_border(&x->block[i+0].bmi.mv.as_mv, x);
                clamp_mv_to_umv_border(&x->block[i+1].bmi.mv.as_mv, x);
            }

            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
            else
            {
                build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
                build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            }

        }

    }
    base_dst = x->dst.u_buffer;
    base_pre = x->pre.u_buffer;
    for (i = 16; i < 20; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];
        int dst_stride = x->dst.uv_stride;

        /* Note: uv mvs already clamped in build_4x4uvmvs() */

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
        else
        {
            build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
        }
    }

    base_dst = x->dst.v_buffer;
    base_pre = x->pre.v_buffer;
    for (i = 20; i < 24; i += 2)
    {
        BLOCKD *d0 = &x->block[i];
        BLOCKD *d1 = &x->block[i+1];
        int dst_stride = x->dst.uv_stride;

        /* Note: uv mvs already clamped in build_4x4uvmvs() */

        if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
            build_inter_predictors2b(x, d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride);
        else
        {
            build_inter_predictors_b(d0, base_dst + d0->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
            build_inter_predictors_b(d1, base_dst + d1->offset, dst_stride, base_pre, dst_stride, x->subpixel_predict);
        }
    }
}

static
void build_4x4uvmvs(MACROBLOCKD *x)
{
    int i, j;

    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 2; j++)
        {
            int yoffset = i * 8 + j * 2;
            int uoffset = 16 + i * 2 + j;
            int voffset = 20 + i * 2 + j;

            int temp;

            temp = x->mode_info_context->bmi[yoffset + 0].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 1].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 4].mv.as_mv.row
                 + x->mode_info_context->bmi[yoffset + 5].mv.as_mv.row;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.row = (temp / 8) & x->fullpixel_mask;

            temp = x->mode_info_context->bmi[yoffset + 0].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 1].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 4].mv.as_mv.col
                 + x->mode_info_context->bmi[yoffset + 5].mv.as_mv.col;

            temp += 4 + ((temp >> (sizeof(temp) * CHAR_BIT - 1)) * 8);

            x->block[uoffset].bmi.mv.as_mv.col = (temp / 8) & x->fullpixel_mask;

            if (x->mode_info_context->mbmi.need_to_clamp_mvs)
                clamp_uvmv_to_umv_border(&x->block[uoffset].bmi.mv.as_mv, x);

            x->block[voffset].bmi.mv.as_int = x->block[uoffset].bmi.mv.as_int;
        }
    }
}

void vp8_build_inter_predictors_mb(MACROBLOCKD *xd)
{
    if (xd->mode_info_context->mbmi.mode != SPLITMV)
    {
        vp8_build_inter16x16_predictors_mb(xd, xd->dst.y_buffer,
                                           xd->dst.u_buffer, xd->dst.v_buffer,
                                           xd->dst.y_stride, xd->dst.uv_stride);
    }
    else
    {
        build_4x4uvmvs(xd);
        build_inter4x4_predictors_mb(xd);
    }
}
