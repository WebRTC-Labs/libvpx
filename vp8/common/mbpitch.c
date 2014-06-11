/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "blockd.h"
#if CONFIG_OPENCL
#include "opencl/vp8_opencl.h"
#endif

void vp8_setup_block_dptrs(MACROBLOCKD *x)
{
    int r, c;

    for (r = 0; r < 4; r++)
    {
        for (c = 0; c < 4; c++)
        {
            x->block[r*4+c].predictor = x->predictor + r * 4 * 16 + c * 4;
        }
    }

    for (r = 0; r < 2; r++)
    {
        for (c = 0; c < 2; c++)
        {
            x->block[16+r*2+c].predictor = x->predictor + 256 + r * 4 * 8 + c * 4;
        }
    }

    for (r = 0; r < 2; r++)
    {
        for (c = 0; c < 2; c++)
        {
            x->block[20+r*2+c].predictor = x->predictor + 320 + r * 4 * 8 + c * 4;
        }
    }

    for (r = 0; r < 25; r++)
    {
        x->block[r].qcoeff  = x->qcoeff  + r * 16;
        x->block[r].dqcoeff = x->dqcoeff + r * 16;
        x->block[r].eob     = x->eobs + r;
#if CONFIG_OPENCL
        if (cl_initialized == CL_SUCCESS){
          /* Copy command queue reference from macroblock */
          x->block[r].cl_commands = x->cl_commands;
          /* Set up CL memory buffers as appropriate */
          x->block[r].cl_dqcoeff_mem = x->cl_dqcoeff_mem;
          x->block[r].cl_eobs_mem = x->cl_eobs_mem;
          x->block[r].cl_predictor_mem = x->cl_predictor_mem;
          x->block[r].cl_qcoeff_mem = x->cl_qcoeff_mem;
        }

        //Copy filter type to block.
        x->block[r].sixtap_filter = x->sixtap_filter;
#endif
    }
}

void vp8_build_block_doffsets(MACROBLOCKD *x)
{
    int block;

    for (block = 0; block < 16; block++) /* y blocks */
    {
        x->block[block].offset =
            (block >> 2) * 4 * x->dst.y_stride + (block & 3) * 4;
    }

    for (block = 16; block < 20; block++) /* U and V blocks */
    {
        x->block[block+4].offset =
        x->block[block].offset =
            ((block - 16) >> 1) * 4 * x->dst.uv_stride + (block & 1) * 4;
    }
}
