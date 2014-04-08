/*
 *  Copyright 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx_scale_rtcd.h"
#include "vp8/decoder/onyxd_int.h"
#include "vp8/common/header.h"
#include "vp8/common/reconintra4x4.h"
#include "vp8/common/reconinter.h"
#include "vp8/decoder/detokenize.h"
#include "vp8/common/invtrans.h"
#include "vp8/common/alloccommon.h"
#include "vp8/common/entropymode.h"
#include "vp8/common/quant_common.h"
#include "vpx_scale/vpx_scale.h"
#include "vp8/common/setupintrarecon.h"

#include "vp8/decoder/decodemv.h"
#include "vp8/common/extend.h"
#if CONFIG_ERROR_CONCEALMENT
#include "vp8/decoder/error_concealment.h"
#endif
#include "vpx_mem/vpx_mem.h"
#include "vp8/common/threading.h"
#include "vp8/decoder/decoderthreading.h"
#include "vp8/decoder/dboolhuff.h"

#include <assert.h>
#include <stdio.h>
#if CONFIG_OPENCL
#include "vp8/common/opencl/vp8_opencl.h"
#include "vp8/common/opencl/blockd_cl.h"
#include "vp8/common/opencl/dequantize_cl.h"
#endif

void mb_init_dequantizer_cl(VP8D_COMP *pbi, MACROBLOCKD *xd)
{
    int i, err;
    //Set up per-block dequant CL memory. Eventually, might be able to set up
    //one large buffer containing the entire large dequant buffer.
    if (cl_initialized == CL_SUCCESS){
        for (i=0; i < 25; i++){
            VP8_CL_CREATE_BUF(xd->cl_commands, xd->block[i].cl_dequant_mem,
                ,
                16*sizeof(cl_short),
                xd->block[i].dequant,,
            );
        }
    }

}

#if CONFIG_RUNTIME_CPU_DETECT
#define RTCD_VTABLE(x) (&(pbi)->common.rtcd.x)
#else
#define RTCD_VTABLE(x) NULL
#endif

void vp8_decode_macroblock_cl(VP8D_COMP *pbi, MACROBLOCKD *xd,
                              unsigned int mb_idx)
{
    MB_PREDICTION_MODE mode = DC_PRED;
    int i;
#if CONFIG_ERROR_CONCEALMENT
    int corruption_detected = 0;
#endif

    /* do prediction */
    if (xd->mode_info_context->mbmi.ref_frame == INTRA_FRAME)
    {
        vp8_build_intra_predictors_mbuv_s(xd,
                                          xd->recon_above[1],
                                          xd->recon_above[2],
                                          xd->recon_left[1],
                                          xd->recon_left[2],
                                          xd->recon_left_stride[1],
                                          xd->dst.u_buffer, xd->dst.v_buffer,
                                          xd->dst.uv_stride);

        if (mode != B_PRED)
        {
            vp8_build_intra_predictors_mby_s(xd,
                                                 xd->recon_above[0],
                                                 xd->recon_left[0],
                                                 xd->recon_left_stride[0],
                                                 xd->dst.y_buffer,
                                                 xd->dst.y_stride);
        }
        else
        {
            short *DQC = xd->dequant_y1;
            int dst_stride = xd->dst.y_stride;

            /* clear out residual eob info */
            if(xd->mode_info_context->mbmi.mb_skip_coeff)
                vpx_memset(xd->eobs, 0, 25);

            intra_prediction_down_copy(xd, xd->recon_above[0] + 16);

            for (i = 0; i < 16; i++)
            {
                BLOCKD *b = &xd->block[i];
                unsigned char *dst = xd->dst.y_buffer + b->offset;
                B_PREDICTION_MODE b_mode =
                    xd->mode_info_context->bmi[i].as_mode;
                unsigned char *Above = dst - dst_stride;
                unsigned char *yleft = dst - 1;
                int left_stride = dst_stride;
                unsigned char top_left = Above[-1];

                vp8_intra4x4_predict(Above, yleft, left_stride, b_mode,
                                     dst, dst_stride, top_left);

                if (xd->eobs[i])
                {
                    if (xd->eobs[i] > 1)
                    {
                    vp8_dequant_idct_add(b->qcoeff, DQC, dst, dst_stride);
                    }
                    else
                    {
                        vp8_dc_only_idct_add
                            (b->qcoeff[0] * DQC[0],
                                dst, dst_stride,
                                dst, dst_stride);
                        vpx_memset(b->qcoeff, 0, 2 * sizeof(b->qcoeff[0]));
                    }
                }
            }
        }
    }
    else
    {
#if ENABLE_CL_SUBPIXEL
        vp8_build_inter_predictors_mb_cl(xd);
#else
        vp8_build_inter_predictors_mb(xd);
#endif

#if (0) //1 || !ENABLE_CL_IDCT_DEQUANT)
        //Wait for inter-predict if dequant/IDCT is being done on the CPU
        VP8_CL_FINISH(xd->cl_commands);
#endif
    }


#if CONFIG_ERROR_CONCEALMENT
    if (corruption_detected)
    {
        return;
    }
#endif

    if(!xd->mode_info_context->mbmi.mb_skip_coeff)
    {
        /* dequantization and idct */
        if (mode != B_PRED)
        {
            short *DQC = xd->dequant_y1;

            if (mode != SPLITMV)
            {
                BLOCKD *b = &xd->block[24];

                /* do 2nd order transform on the dc block */
                if (xd->eobs[24] > 1)
                {
                    vp8_dequantize_b(b, xd->dequant_y2);

                    vp8_short_inv_walsh4x4(&b->dqcoeff[0],
                        xd->qcoeff);
                    vpx_memset(b->qcoeff, 0, 16 * sizeof(b->qcoeff[0]));
                }
                else
                {
                    b->dqcoeff[0] = b->qcoeff[0] * xd->dequant_y2[0];
                    vp8_short_inv_walsh4x4_1(&b->dqcoeff[0],
                        xd->qcoeff);
                    vpx_memset(b->qcoeff, 0, 2 * sizeof(b->qcoeff[0]));
                }

                /* override the dc dequant constant in order to preserve the
                 * dc components
                 */
                DQC = xd->dequant_y1_dc;
            }

            vp8_dequant_idct_add_y_block
                            (xd->qcoeff, DQC,
                             xd->dst.y_buffer,
                             xd->dst.y_stride, xd->eobs);
        }

        vp8_dequant_idct_add_uv_block
                        (xd->qcoeff+16*16, xd->dequant_uv,
                         xd->dst.u_buffer, xd->dst.v_buffer,
                         xd->dst.uv_stride, xd->eobs+16);
    }
}

void vp8_decode_frame_cl_finish(VP8D_COMP *pbi){

    //If using OpenCL, free all of the GPU buffers we've allocated.
    if (cl_initialized == CL_SUCCESS){
        //Wait for stuff to finish, just in case
        VP8_CL_FINISH(pbi->mb.cl_commands);

#if !ONE_CQ_PER_MB
        VP8_CL_FINISH(pbi->mb.block[0].cl_commands);
        VP8_CL_FINISH(pbi->mb.block[16].cl_commands);
        VP8_CL_FINISH(pbi->mb.block[20].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[0].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[16].cl_commands);
        clReleaseCommandQueue(pbi->mb.block[20].cl_commands);
#endif

#if ENABLE_CL_IDCT_DEQUANT || ENABLE_CL_SUBPIXEL
        //Free Predictor CL buffer
        if (pbi->mb.cl_predictor_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_predictor_mem);
#endif

#if ENABLE_CL_IDCT_DEQUANT
        //Free other CL Block/MBlock buffers
        if (pbi->mb.cl_qcoeff_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_qcoeff_mem);
        if (pbi->mb.cl_dqcoeff_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_dqcoeff_mem);
        if (pbi->mb.cl_eobs_mem != NULL)
            clReleaseMemObject(pbi->mb.cl_eobs_mem);

        for (int i = 0; i < 25; i++){
            clReleaseMemObject(pbi->mb.block[i].cl_dequant_mem);
            pbi->mb.block[i].cl_dequant_mem = NULL;
        }
#endif
    }
}
