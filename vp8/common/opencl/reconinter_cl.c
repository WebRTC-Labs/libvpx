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

static void vp8_build_inter_predictors_b_cl(
    MACROBLOCKD *x, BLOCKD *d, int pitch, unsigned char *base_pre, int pre_stride) {
  unsigned char *ptr_base = *(base_pre);
  int ptr_offset = d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

  vp8_subpix_cl_fn_t sppf;

  int pre_dist = base_pre - x->pre.buffer_alloc;
  cl_mem pre_mem = x->pre.buffer_mem;
  int pre_off = pre_dist+ptr_offset;

  if (d->sixtap_filter == CL_TRUE)
      sppf = vp8_sixtap_predict4x4_cl;
  else
      sppf = vp8_bilinear_predict4x4_cl;

  if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7) {
    sppf(d->cl_commands, ptr_base, pre_mem, pre_off, pre_stride,
         d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7, d->predictor,
         d->cl_predictor_mem, 0, pitch);
  } else {
    int pred_off = 0;
    vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, pre_stride,
          d->cl_predictor_mem, &pred_off, pitch,4,4,1);
  }
}


static void vp8_build_inter_predictors4b_cl(
    MACROBLOCKD *x, BLOCKD *d, int pitch, unsigned char *base_pre, int pre_stride) {
  unsigned char *ptr_base = *(base_pre);
  int ptr_offset = d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

  int pre_dist = base_pre - x->pre.buffer_alloc;
  cl_mem pre_mem = x->pre.buffer_mem;
  int pre_off = pre_dist + ptr_offset;

  //If there's motion in the bottom 8 subpixels, need to do subpixel prediction
  if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7) {
    if (d->sixtap_filter == CL_TRUE) {
      vp8_sixtap_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off,
          pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7,
          d->predictor, d->cl_predictor_mem, 0, pitch);
    } else {
      vp8_bilinear_predict8x8_cl(d->cl_commands, ptr_base, pre_mem, pre_off,
          pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7,
          d->predictor, d->cl_predictor_mem, 0, pitch);
   }
  } else { //Otherwise copy memory directly from src to dest
    int pred_off = 0;
    vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, pre_stride,
        d->cl_predictor_mem, &pred_off, pitch, 8, 8, 1);
  }
}

static void vp8_build_inter_predictors2b_cl(
    MACROBLOCKD *x, BLOCKD *d, int pitch, unsigned char *base_pre, int pre_stride) {
  unsigned char *ptr_base = *(base_pre);

  int ptr_offset = d->offset + (d->bmi.mv.as_mv.row >> 3) * pre_stride + (d->bmi.mv.as_mv.col >> 3);

  int pre_dist = base_pre - x->pre.buffer_alloc;
  cl_mem pre_mem = x->pre.buffer_mem;
  int pre_off = pre_dist+ptr_offset;

  if ( (d->bmi.mv.as_mv.row | d->bmi.mv.as_mv.col) & 7) {
    if (d->sixtap_filter == CL_TRUE) {
      vp8_sixtap_predict8x4_cl(d->cl_commands,ptr_base,pre_mem,pre_off,
        pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7,
        d->predictor, d->cl_predictor_mem, 0, pitch);
    } else {
      vp8_bilinear_predict8x4_cl(d->cl_commands,ptr_base,pre_mem,pre_off,
        pre_stride, d->bmi.mv.as_mv.col & 7, d->bmi.mv.as_mv.row & 7,
        d->predictor, d->cl_predictor_mem, 0, pitch);
    }
  } else {
    int pred_off = 0;
    vp8_copy_mem_cl(d->cl_commands, pre_mem, &pre_off, pre_stride,
        d->cl_predictor_mem, &pred_off, pitch, 8, 4, 1);
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
        int pre_stride = x->pre.uv_stride;

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
        uint8_t *base_pre;
        int pre_stride = x->pre.uv_stride;
        for (i = 16; i < 24; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];
            base_pre = (i < 20) ? x->pre.u_buffer : x->pre.v_buffer;  // mcs
            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                vp8_build_inter_predictors2b_cl(x, d0, 8, base_pre, pre_stride);
            else
            {
                vp8_build_inter_predictors_b_cl(x, d0, 8, base_pre, pre_stride);
                vp8_build_inter_predictors_b_cl(x, d1, 8, base_pre, pre_stride);
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
        int pre_stride = x->pre.y_stride;

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
        uint8_t *base_pre = x->pre.y_buffer;
        int pre_stride = x->pre.y_stride;

        if (x->mode_info_context->mbmi.partitioning < 3) {
            for (i = 0; i < 4; i++)
            {
                BLOCKD *d = &x->block[bbb[i]];
                vp8_build_inter_predictors4b_cl(x, d, 16, base_pre, pre_stride);
            }
        } else {
            /* This loop can be done in any order... No dependencies.*/
            /* Also, d0/d1 can be decoded simultaneously */
            for (i = 0; i < 16; i += 2) {
                BLOCKD *d0 = &x->block[i];
                BLOCKD *d1 = &x->block[i+1];

                if (d0->bmi.mv.as_int == d1->bmi.mv.as_int)
                    vp8_build_inter_predictors2b_cl(x, d0, 16, base_pre, pre_stride);
                else
                {
                    vp8_build_inter_predictors_b_cl(x, d0, 16, base_pre, pre_stride);
                    vp8_build_inter_predictors_b_cl(x, d1, 16, base_pre, pre_stride);
                }
            }
        }

        /* Another case of re-orderable/batchable loop */
        for (i = 16; i < 24; i += 2)
        {
            BLOCKD *d0 = &x->block[i];
            BLOCKD *d1 = &x->block[i+1];
            base_pre = (i < 20) ? x->pre.u_buffer : x->pre.v_buffer;  // mcs
            pre_stride = x->pre.uv_stride;  // mcs

            if (d0->bmi.mv.as_int == d1->bmi.mv.as_int) {
              vp8_build_inter_predictors2b_cl(x, d0, 8, base_pre, pre_stride);
            } else {
              vp8_build_inter_predictors_b_cl(x, d0, 8, base_pre, pre_stride);
              vp8_build_inter_predictors_b_cl(x, d1, 8, base_pre, pre_stride);
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

