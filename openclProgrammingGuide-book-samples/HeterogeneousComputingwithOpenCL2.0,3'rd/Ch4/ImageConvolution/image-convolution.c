#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

#ifdef __cplusplus
extern "C" {
#endif

#define  CL_FILE_NAME   "./HeterogeneousComputingwithOpenCL2.0,3'rd/Ch4/ImageConvolution/image-convolution.cl"
	
static const char* inputImagePath = "./HeterogeneousComputingwithOpenCL2.0,3'rd/Images/cat.bmp";
static float gaussianBlurFilterFactor = 273.0f;
static float gaussianBlurFilter[25] = {
	   1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
	   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
	   7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
	   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
	   1.0f,  4.0f,  7.0f,  4.0f, 1.0f };
	static const int gaussianBlurFilterWidth = 5;

	static float sharpenFilterFactor = 8.0f;
	static float sharpenFilter[25] = {
		-1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
		-1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
		-1.0f,  2.0f,  8.0f,  2.0f, -1.0f,
		-1.0f,  2.0f,  2.0f,  2.0f, -1.0f,
		-1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
	static const int sharpenFilterWidth = 5;

	static int edgeSharpenFilterFactor = 1.0f;
	static int edgeSharpenFilter[9] = {
		1.0f,  1.0f, 1.0f,
		1.0f, -7.0f, 1.0f,
		1.0f,  1.0f, 1.0f };
	static const int edgeSharpenFilterWidth = 3;

	static int vertEdgeDetectFilterFactor = 1.0f;
	static int vertEdgeDetectFilter[25] = {
		 0,  0, -1.0f,  0,  0,
		 0,  0, -1.0f,  0,  0,
		 0,  0,  4.0f,  0,  0,
		 0,  0, -1.0f,  0,  0,
		 0,  0, -1.0f,  0,  0 };
	static const int vertEdgeDetectFilterWidth = 3;

	static float embossFilterFactor = 1.0f;
	static float embossFilter[9] = {
		2.0f,  0.0f,  0.0f,
		0.0f, -1.0f,  0.0f,
		0.0f,  0.0f, -1.0f };
	static const int embossFilterWidth = 3;

	enum filterList
	{
		GAUSSIAN_BLUR,
		SHARPEN,
		EDGE_SHARPEN,
		VERT_EDGE_DETECT,
		EMBOSS,
		FILTER_LIST_SIZE
	};
	static const int filterSelection = VERT_EDGE_DETECT;

	int main(int argc, char **argv)
	{
		int i;
		cl_int status = CL_FALSE;
		cl_kernel kernel = NULL;
		cl_platform_id platform = NULL;
		cl_device_id device = NULL;
		cl_context context = NULL;
		cl_program program = NULL;
		int *hInputImage = NULL;
		int *hOutputImage = NULL;
		cl_mem inputImage, outputImage;
		int passed = TRUE;
		cl_sampler sampler;
		int imageRows;
		int imageCols;
		cl_image_format fmt;
		cl_image_desc desc;
		int filterWidth;
		float filterFactor;
		float *filter;

		//filter mode
		switch (filterSelection)
		{
		case GAUSSIAN_BLUR:
			filterWidth = gaussianBlurFilterWidth;
			filterFactor = gaussianBlurFilterFactor;
			filter = gaussianBlurFilter;
			break;
		case SHARPEN:
			filterWidth = sharpenFilterWidth;
			filterFactor = sharpenFilterFactor;
			filter = sharpenFilter;
			break;
		case EDGE_SHARPEN:
			filterWidth = edgeSharpenFilterWidth;
			filterFactor = edgeSharpenFilterFactor;
			filter = edgeSharpenFilter;
			break;
		case VERT_EDGE_DETECT:
			filterWidth = vertEdgeDetectFilterWidth;
			filterFactor = vertEdgeDetectFilterFactor;
			filter = vertEdgeDetectFilter;
			break;
		case EMBOSS:
			filterWidth = embossFilterWidth;
			filterFactor = embossFilterFactor;
			filter = embossFilter;
			break;
		default:
			printf("Invalid filter selection\n");
			return 1;
		}

		for (int i = 0; i < filterWidth*filterWidth; i++)
		{
			filter[i] = filter[i] / filterFactor;
		}

		//initial 
		//hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
		hInputImage = readBmp(inputImagePath, &imageRows, &imageCols);

		hOutputImage = (float*)malloc(imageRows*imageCols * sizeof(float));
		if (hOutputImage == NULL) {
			printf("malloc hOutputImage failed\n");
		}

		OpenCLInit(&platform, &device, &context);
		printDevInfo(platform, device);

		// Create the images 
		fmt.image_channel_order = CL_R;
		fmt.image_channel_data_type = CL_SIGNED_INT32;

		memset(&desc, '\0', sizeof(cl_image_desc));
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		//desc.image_depth = 8;
		//desc.image_slice_pitch = 0;
		desc.image_width = imageCols ;
		desc.image_height = imageRows;
		desc.buffer = NULL; // or someBuf;
		
		// Create Image Object 
		inputImage = clCreateImage(context    /* context */,
			CL_MEM_READ_WRITE            /* flags */,
			&fmt                      /* image_format */,
			&desc			        /* image_desc */,
			NULL                  /* host_ptr */,
			&status                /* errcode_ret */);
		check(status);
		outputImage = clCreateImage(context,   
			CL_MEM_READ_WRITE            /* flags */,
			&fmt                      /* image_format */,
			&desc			        /* image_desc */,
			NULL                  /* host_ptr */,
			&status                /* errcode_ret */);
		check(status);

		// Create a buffer for the filter 
		cl_mem filterBuffer;
		filterBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, filterWidth*filterWidth * sizeof(float), NULL,
			&status);
		check(status);

		cl_command_queue cmdQueue;
		cmdQueue = clCreateCommandQueue(context, device, 0, &status);
		check(status);

		//copy input data to the image,init queue's image
		size_t origin[3] = {0,0,0};
		size_t region[3] = { imageRows, imageCols,1};

		status = clEnqueueWriteImage(cmdQueue    /* command_queue */,
			inputImage              /* image */,
			CL_TRUE             /* blocking_write */,
			origin      /* origin[3] */,
			region     /* region[3] */,
			imageRows  *imageCols* sizeof(float)              /* input_row_pitch */,
			0             /* input_slice_pitch */,
			hInputImage        /* ptr */,
			0             /* num_events_in_wait_list */,
			NULL   /* event_wait_list */,
			NULL          /* event */);

		check(status);  
		status = clEnqueueWriteImage(cmdQueue    /* command_queue */,
			outputImage              /* image */,
			CL_TRUE             /* blocking_write */,
			origin      /* origin[3] */,
			region     /* region[3] */,
			imageRows * imageCols * sizeof(float)              /* input_row_pitch */,
			0             /* input_slice_pitch */,
			hOutputImage        /* ptr */,
			NULL             /* num_events_in_wait_list */,
			NULL   /* event_wait_list */,
			NULL          /* event */) CL_API_SUFFIX__VERSION_1_0;
		check(status);		

		//init queue's buf
		status = clEnqueueWriteBuffer(cmdQueue, filterBuffer, CL_TRUE, 0, filterWidth*filterWidth * sizeof(float),
			(void *)filter, 0, NULL, NULL);
		check(status);

		sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST,&status);
		check(status);
		status = BuildKernel(&device, &context, &program, CL_FILE_NAME);
		check(status);
		kernel = clCreateKernel(program, "convolution", &status);
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
		status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
		status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
		status |= clSetKernelArg(kernel, 3, sizeof(cl_int), &filterWidth);
		status |= clSetKernelArg(kernel, 4, sizeof(cl_sampler), &sampler);
		check(status);

		// Define the index space and work-group size
		size_t globalWorkSize[2] = { imageCols, imageRows };

		size_t localWorkSize[2] = { 8,8 };
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, &globalWorkSize[0], &localWorkSize[0], 0, NULL, NULL);
		check(status);

		
#if 0
		// Execute the kernel
		cl::NDRange global(imageCols, imageRows);
		cl::NDRange local(8, 8);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

		/* Copy the output data back to the host */
		queue.enqueueReadImage(outputImage, CL_TRUE, origin, region, 0, 0,
			hOutputImage);
#endif
		// Save the output bmp 
		printf("Output image saved as: cat-filtered.bmp\n");
		writeBmpFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols,
			inputImagePath);

		/* Verify result */
		int *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
			filter, filterWidth);

		for (i = 0; i < imageRows*imageCols; i++) {
			if (fabs(refOutput[i] - hOutputImage[i]) > 0.001f) {
				passed = FALSE;
			}
		}
		if (passed) {
			printf("Passed!\n");
		}
		else {
			printf("failed!\n");
		}
		free(refOutput);
		free(hInputImage);
		free(hOutputImage);
		char c = getchar();
		return 0;
	}

#ifdef __cplusplus
}
#endif