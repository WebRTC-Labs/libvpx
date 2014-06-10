#include "vpx_config.h"

#include "vp8/common/opencl/vp8_opencl.h"
#include "vp8_decode_cl.h"

#include <stdio.h>

extern int cl_init_dequant();
extern int cl_destroy_dequant();

int cl_decode_destroy(){
  return CL_SUCCESS;
}

int cl_decode_init()
{
  //Initialize programs to null value
  //Enables detection of if they've been initialized as well.
  cl_data.dequant_program = NULL;
  return CL_SUCCESS;
}
