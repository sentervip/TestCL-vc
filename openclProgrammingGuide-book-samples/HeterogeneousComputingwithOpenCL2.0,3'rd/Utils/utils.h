#ifndef __UTILS_H__
#define __UTILS_H__

/* OpenCL includes */
#include <CL/cl.h>
#define  TRUE   1
#define  FALSE  0

void check(cl_int);
void printCompilerError(cl_program program, cl_device_id device);
int readFile2(const char *filename, char * outFileData);
char* readFile(const char *filename);
int BuildKernel(cl_device_id* device, cl_context* context, cl_program *program, const char* fileName);
void OpenCLInit(cl_platform_id *clPlatform, cl_device_id *clDevice, cl_context *clContext);
int printDevInfo(cl_platform_id platform, cl_device_id device);
#endif
