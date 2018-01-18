/* System includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OpenCL includes */
#include <CL/cl.h>

/* Utility functions */
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

#define  fileName   "./HeterogeneousComputingwithOpenCL2.0,3'rd/Ch4/Histogram/histogram.cl"
static const int HIST_BINS = 256; 

int main(int argc, char **argv) 
{
   cl_int status = CL_FALSE;
   cl_kernel kernel = NULL;
   cl_platform_id platform = NULL;
   cl_device_id device = NULL;
   cl_context context = NULL;
   cl_program program = NULL;
   int *hInputImage = NULL; 
   int *hOutputHistogram = NULL;
   int imageRows = 0;
   int imageCols = 0;
   hInputImage = readBmp("./HeterogeneousComputingwithOpenCL2.0,3'rd/Images/cat.bmp",
	                      &imageRows, &imageCols);
   const int imageElements = imageRows*imageCols;
   const size_t imageSize = imageElements*sizeof(int);

   /* Allocate space for the histogram on the host */
   const int histogramSize = HIST_BINS*sizeof(int);
   hOutputHistogram = (int*)malloc(histogramSize);
   if (!hOutputHistogram) { 
	   exit(-1);
   }

   OpenCLInit(&platform, &device, &context);
   
   /* Create a command queue and associate it with the device */
   cl_command_queue cmdQueue;
  // cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);//CL2.0
   cmdQueue = clCreateCommandQueue(context, device, 0, &status);
   check(status);

   /* Create a buffer object for the input image */
   cl_mem bufInputImage;
   bufInputImage = clCreateBuffer(context, CL_MEM_READ_ONLY, imageSize, NULL, 
                                  &status);
                                  check(status);

   /* Create a buffer object for the output histogram */
   cl_mem bufOutputHistogram;
   bufOutputHistogram = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
      histogramSize, NULL, &status);
   check(status);

   /* Write the input image to the device */
   status = clEnqueueWriteBuffer(cmdQueue, bufInputImage, CL_TRUE, 0, imageSize,
         hInputImage, 0, NULL, NULL);
   check(status);

   /* Initialize the output histogram with zeros */
   int zero = 0;
   status = clEnqueueFillBuffer(cmdQueue, bufOutputHistogram, &zero, 
         sizeof(int), 0, histogramSize, 0, NULL, NULL);
   check(status);

   //build kerenl
   BuildKernel(&device, &context, &program, fileName);
   kernel = clCreateKernel(program, "histogram", &status);
   status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufInputImage);
   status |= clSetKernelArg(kernel, 1, sizeof(int), &imageElements);
   status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOutputHistogram);
   check(status);

   /* Define the index space and work-group size */
   size_t globalWorkSize[1];
   globalWorkSize[0] = 1024;

   size_t localWorkSize[1];
   localWorkSize[0] = 64;

   /* Enqueue the kernel for execution */
   status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
      globalWorkSize, localWorkSize, 0, NULL, NULL);
   check(status);

   /* Read the output histogram buffer to the host */
   status = clEnqueueReadBuffer(cmdQueue, bufOutputHistogram, CL_TRUE, 0,
         histogramSize, hOutputHistogram, 0, NULL, NULL);
   check(status);

   /* Verify the output */
   int *refHistogram;
   refHistogram = histogramGold(hInputImage, imageRows*imageCols, HIST_BINS);
   int passed = 1;
   int i;
   for (i = 0; i < HIST_BINS; i++) {
      if (hOutputHistogram[i] != refHistogram[i]) {
         passed = 0;
      }
   }
   if (passed) {
      printf("Passed!\n");
   }
   else {
      printf("Failed.\n");
   }
   free(refHistogram);

   /* Free OpenCL resources */
   clReleaseKernel(kernel);
   clReleaseProgram(program);
   clReleaseCommandQueue(cmdQueue);
   clReleaseMemObject(bufInputImage);
   clReleaseMemObject(bufOutputHistogram);
   clReleaseContext(context);

   /* Free host resources */
   free(hInputImage);
   free(hOutputHistogram);
   char c = getchar();
   return 0;
}
