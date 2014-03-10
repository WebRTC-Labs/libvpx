/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "vp8_opencl.h"

int cl_initialized = VP8_CL_NOT_INITIALIZED;
VP8_COMMON_CL cl_data;

//Initialization functions for various CL programs.
extern int cl_init_filter();
extern int cl_init_idct();
extern int cl_init_loop_filter();

//Common CL destructors
extern void cl_destroy_loop_filter();
extern void cl_destroy_filter();
extern void cl_destroy_idct();

//Destructors for encoder/decoder-specific bits
extern void cl_decode_destroy();
extern void cl_encode_destroy();

/**
 *
 * @param cq
 * @param new_status
 */
void cl_destroy(cl_command_queue cq, int new_status) {

  if (cl_initialized != CL_SUCCESS)
    return;

    //Wait on any pending operations to complete... frees up all of our pointers
  if (cq != NULL)
    VP8_CL_FINISH(cq);

#if ENABLE_CL_SUBPIXEL
    //Release the objects that we've allocated on the GPU
  cl_destroy_filter();
#endif

#if ENABLE_CL_IDCT_DEQUANT
  cl_destroy_idct();

#if CONFIG_VP8_DECODER
  if (cl_data.cl_decode_initialized == CL_SUCCESS)
    cl_decode_destroy();
#endif

#endif
#if ENABLE_CL_LOOPFILTER
  cl_destroy_loop_filter();
#endif


#if CONFIG_VP8_ENCODER
    //placeholder for if/when encoder CL gets implemented
#endif

  if (cq){
    clReleaseCommandQueue(cq);
  }

  if (cl_data.context){
    clReleaseContext(cl_data.context);
    cl_data.context = NULL;
  }

  cl_initialized = new_status;

  return;
}

/**
 *
 * @param dev
 * @return
 */
cl_device_type device_type(cl_device_id dev){
  cl_device_type type;
  int err;

  err = clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type),&type,NULL);
  if (err != CL_SUCCESS)
    return CL_INVALID_DEVICE;
  return type;
}

/**
 *
 * @return
 */
int cl_common_init() {
  int err,i,dev;
  cl_platform_id platform_ids[MAX_NUM_PLATFORMS];
  cl_uint num_found, num_devices;
  cl_device_id devices[MAX_NUM_DEVICES];

  //Don't allow multiple CL contexts..
  if (cl_initialized != VP8_CL_NOT_INITIALIZED)
    return cl_initialized;

  // Connect to a compute device
  err = clGetPlatformIDs(MAX_NUM_PLATFORMS, platform_ids, &num_found);

  if (err != CL_SUCCESS) {
    fprintf(stderr, "Couldn't query platform IDs\n");
    return VP8_CL_TRIED_BUT_FAILED;
  }

  if (num_found == 0) {
    fprintf(stderr, "No platforms found\n");
    return VP8_CL_TRIED_BUT_FAILED;
  }

  //printf("Enumerating %d platform(s)\n", num_found);
  //Enumerate the platforms found. Platforms are like Apple, Windows, etc.
  for (i = 0; i < num_found; i++){
    char buf[2048];
    size_t len;

    err = clGetPlatformInfo( platform_ids[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, &len);
    if (err != CL_SUCCESS){
      fprintf(stderr, "Error retrieving platform vendor for platform %d",i);
      continue;
    }
    printf("Platform %d: %s\n",i,buf);

    err = clGetPlatformInfo( platform_ids[i], CL_PLATFORM_VERSION, sizeof(buf), buf, &len);
    if (err != CL_SUCCESS){
      fprintf(stderr, "Error retrieving platform version for platform %d",i);
      continue;
    }
    printf("Version %d: %s\n",i,buf);

    // Try to find a valid compute device
    // Favor the GPU, but fall back to any other available device if necessary
#if defined(__MACOS_10_6__)
    printf("Running CL as CPU-only for now...\n");
    err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_CPU, MAX_NUM_DEVICES, devices, &num_devices);
#else
    err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, MAX_NUM_DEVICES, devices, &num_devices);
#endif //Snow Leopard
    printf("Found %d devices:\n", num_devices);
    cl_data.device_id = NULL;
    for( dev = 0; dev < num_devices; dev++ ){
#if ENABLE_CL_IDCT_DEQUANT == 1 || ENABLE_CL_SUBPIXEL == 1
      char ext[2048];

      // Get info for this device.
      err = clGetDeviceInfo(devices[dev], CL_DEVICE_EXTENSIONS,
        sizeof(ext),ext,NULL);
      VP8_CL_CHECK_SUCCESS(NULL,err != CL_SUCCESS,
        "Error retrieving device extension list",continue, 0);

      // printf("Device %d supports: %s\n",dev,ext);
      // The prediction/IDCT kernels in VP8 require byte-addressable stores,
      // which is an extension. It's required in OpenCL 1.1, but not all devices
      // support that version.
      if (strstr(ext,"cl_khr_byte_addressable_store")){
#endif
        char* value;
        size_t valueSize;

        clGetDeviceInfo(devices[dev], CL_DEVICE_NAME, 0, NULL, &valueSize);
        value = (char*) malloc(valueSize);
        clGetDeviceInfo(devices[dev], CL_DEVICE_NAME, valueSize, value, NULL);
        printf("%d. Device: %s\n", dev, value);
        free(value);

        cl_data.device_id = devices[dev];
        cl_data.device_type = device_type(devices[dev]);
        // Prefer using a GPU.
        if ( cl_data.device_type == CL_DEVICE_TYPE_GPU ){
          printf("Device %d is a GPU, stop searching.\n",dev);
          break;
        }
#if ENABLE_CL_IDCT_DEQUANT == 1 || ENABLE_CL_SUBPIXEL == 1
      }
#endif
    }

    // If we've found a usable GPU, stop looking.
    if (cl_data.device_id != NULL && cl_data.device_type == CL_DEVICE_TYPE_GPU )
      break;

  }

  if (cl_data.device_id == NULL){
    printf("Error: Failed to find a valid OpenCL device. Using CPU paths\n");
    return VP8_CL_TRIED_BUT_FAILED;
  }

    // Create the compute context
  cl_data.context = clCreateContext(0, 1, &cl_data.device_id, NULL, NULL, &err);
  if (!cl_data.context) {
    printf("Error: Failed to create a compute context!\n");
    return VP8_CL_TRIED_BUT_FAILED;
  }

  // Initialize programs to null value
  // Enables detection of if they've been initialized as well.
  cl_data.filter_program = NULL;
  cl_data.idct_program = NULL;
  cl_data.loop_filter_program = NULL;

#if ENABLE_CL_SUBPIXEL
  err = cl_init_filter();
  if (err != CL_SUCCESS)
    return err;
#endif

#if ENABLE_CL_IDCT_DEQUANT
  err = cl_init_idct();
  if (err != CL_SUCCESS)
    return err;
#endif

#if ENABLE_CL_LOOPFILTER
  err = cl_init_loop_filter();
  if (err != CL_SUCCESS)
    return err;
#endif

  return CL_SUCCESS;
}

//Allocates and returns the full file path for the requested file
char *cl_get_file_path(const char* file_name, char *ext){
  char *fullpath;
  FILE *f;

  fullpath = malloc(strlen(file_name) + strlen(ext) + 1);
  if (fullpath == NULL){
    return NULL;
  }

  strcpy(fullpath, file_name);
  strcat(fullpath, ext);

  f = fopen(fullpath, "rb");
  if (f != NULL){
    fclose(f);
    return fullpath;
  }

  free(fullpath);

    //Generate a file path for the CL sources using the library install dir
  fullpath = malloc(strlen(vpx_codec_lib_dir()) + strlen(file_name) + strlen(ext) + 2);
  if (fullpath == NULL) {
    return NULL;
  }
  strcpy(fullpath, vpx_codec_lib_dir());
  strcat(fullpath, "/"); //Will need to be changed for MSVS
  strcat(fullpath, file_name);
  strcat(fullpath, ext);

  f = fopen(fullpath, "rb");
  if (f == NULL) {
    free(fullpath);
    return NULL;
  }

  return fullpath;
}

char *cl_read_file(const char* file_name, int pad, size_t *size) {
  long pos;
  char *bytes;
  size_t amt_read;
  FILE *f;
   *size = 0;
   if (file_name == NULL){
    return NULL;
  }

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Error opening file %s\n", file_name);
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  pos = ftell(f);
  fseek(f, 0, SEEK_SET);
  *size = pos + pad;
  bytes = malloc(*size);
   if (bytes == NULL) {
    fclose(f);
    return NULL;
  }

  amt_read = fread(bytes, pos, 1, f);
  if (amt_read != 1) {
    free(bytes);
    fclose(f);
    return NULL;
  }

  if (pad > 0){
    int i;
    for (i = 0; i < pad; i++){
      bytes[pos+i] = '\0'; // null terminate the source string
    }
  }
  fclose(f);
  return bytes;
}

char *cl_read_source_file(const char* file_name, size_t *size) {
  return cl_read_file(file_name, 1, size);
}

char *cl_read_binary_file(const char* file_name, size_t *size){
  return cl_read_file(file_name, 0, size);
}

void show_build_log(cl_program *prog_ref){
  size_t len;
  char *buffer;
  int err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

  if (err != CL_SUCCESS){
    printf("Error: Could not get length of CL build log\n");
  }

  buffer = (char*) malloc(len);
  if (buffer == NULL) {
    printf("Error: Couldn't allocate compile output buffer memory\n");
    return;
  }

  err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Could not get CL build log\n");
  } else {
    int err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_OPTIONS, 0, NULL, &len);
    if (err == CL_SUCCESS){
      char *opts = malloc(len);
      if (opts != NULL){
        err = clGetProgramBuildInfo(*prog_ref, cl_data.device_id, CL_PROGRAM_BUILD_OPTIONS, len, opts, NULL);
        if (err == CL_SUCCESS){
          printf("Program build options:\n%s\n", opts);
        }
        free(opts);
      }
    }

    printf("Compile output:\n%s\n", buffer);
  }
  free(buffer);
}

cl_int vp8_cl_save_binary(const char *file_name, const char *ext, cl_program *prog_ref, const char *prog_opts){
  int err;
  char *binary;
  size_t size;
  FILE *out;

  char *bin_file = malloc(strlen(file_name)+strlen(ext)+1);
  if (bin_file == NULL){
    return VP8_CL_TRIED_BUT_FAILED;
  }
  strcpy(bin_file, file_name);
  strcat(bin_file, ext);

  err = clBuildProgram(*prog_ref, 0, NULL, prog_opts, NULL, NULL);
  clGetProgramInfo( *prog_ref, CL_PROGRAM_BINARY_SIZES, sizeof(size), &size,
    NULL );

  binary = (char *)malloc(size);
  if (binary == NULL){
    free(bin_file);
    printf("Couldn't allocate memory to save binary kernel\n");
    return VP8_CL_TRIED_BUT_FAILED;
  }

  err = clGetProgramInfo( *prog_ref, CL_PROGRAM_BINARIES, sizeof(char**), &binary, NULL );
  VP8_CL_CHECK_SUCCESS(NULL, err != CL_SUCCESS, "Couldn't get program info\n", , VP8_CL_TRIED_BUT_FAILED);

  out = fopen( bin_file, "wb" );
  if (out != NULL){
    fwrite( binary, sizeof(char), size, out );
    fclose( out );
  } else {
        //printf("Couldn't save binary kernel\n");
  }

  free(bin_file);

  return CL_SUCCESS;
}

int cl_use_binary_kernel(char *src_file, char *bin_file){
  int ret;

  struct stat src_stat;
  struct stat bin_stat;

  ret = (bin_file != NULL);
  if (bin_file == NULL){
    return ret;
  }

  //Stat the two files. If either fails, use current return code
  //Note: Useful if you have source or binary kernels, but not both
  if (stat(src_file, &src_stat) || stat(bin_file, &bin_stat)){
    return ret;
  }

  //Get the modified date for each, and make sure that binary is newer than src
#ifdef __APPLE__
  ret = bin_stat.st_mtime > src_stat.st_mtime;
#else
  ret = bin_stat.st_mtim.tv_sec > src_stat.st_mtim.tv_sec;
#endif

  return ret;
}

int cl_load_program(cl_program *prog_ref, const char *file_name, const char *opts) {

  int err;
  char *src_file = NULL;
  char *bin_file = NULL;
  char *kernel_src;
  char *kernel_bin;
    size_t size; //Size of loaded kernel src/bin file

    *prog_ref = NULL;

    src_file = cl_get_file_path(file_name, ".cl");
    bin_file = cl_get_file_path(file_name, ".bin");

    if (cl_use_binary_kernel(src_file, bin_file)){
      // Attempt to load binary kernel first
      kernel_bin = cl_read_binary_file(bin_file, &size);
      if (kernel_bin != NULL){
        cl_int status;
        *prog_ref = clCreateProgramWithBinary(cl_data.context, 1, &(cl_data.device_id), &size, (const unsigned char**)&kernel_bin, &status, &err);
        if (status == CL_SUCCESS && err == CL_SUCCESS){
          err = clBuildProgram(*prog_ref, 0, NULL, opts, NULL, NULL);
          if (err != CL_SUCCESS || *prog_ref == NULL){
                    //This block gets executed if binary was for wrong device type
            clReleaseProgram(*prog_ref);
            *prog_ref = NULL;
          } else {
                    //Binary loaded successfully. Free bin_file.
            free(bin_file);
          }
        } else {
          if (*prog_ref != NULL){
            //The loaded binary was not a valid program at all...
            //This might mean it was from a different platform?
            //printf("Failed to create program from binary\n");
            clReleaseProgram(*prog_ref);
            *prog_ref = NULL;
          }
        }
        free(kernel_bin);
      }
    }

    //Binary kernel failed, compile source instead
    if (*prog_ref == NULL){
      kernel_src = cl_read_source_file(src_file, &size);
      free(src_file);

      if (kernel_src != NULL) {
        const char *src = kernel_src;
        *prog_ref = clCreateProgramWithSource(cl_data.context, 1, &src, NULL, &err);
        free(kernel_src);

        if (err != CL_SUCCESS) {
          printf("Error creating program: %d\n", err);
        }

        //Attempt to save program binary
        if (*prog_ref != NULL){
          //vp8_cl_save_binary(file_name, ".bin", prog_ref, opts);
          //free(bin_file);

          // Build the program executable
          err = clBuildProgram(*prog_ref, 0, NULL, opts, NULL, NULL);
          if (err != CL_SUCCESS) {
            printf("Error: Failed to build program executable for %s!\n", file_name);
            show_build_log(prog_ref);
            return VP8_CL_TRIED_BUT_FAILED;
          }
        } else {
          printf("Error: Couldn't create program\n");
          free(bin_file);
          return VP8_CL_TRIED_BUT_FAILED;
        }
      } else {
        cl_destroy(NULL, VP8_CL_TRIED_BUT_FAILED);
        free(bin_file);
        printf("Couldn't find OpenCL source files. \nUsing software path.\n");
        return VP8_CL_TRIED_BUT_FAILED;
      }
    }

    return CL_SUCCESS;
  }
