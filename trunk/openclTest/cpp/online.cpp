#include <stdio.h>
#include <stdlib.h>
//#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (10)
#define MAX_SOURCE_SIZE (0x100000)

int printDevInfo(cl_platform_id platform, cl_device_id device);
int main_online()
{
	cl_uint					numPlatforms;
	cl_platform_id			platform;
	cl_platform_id			*platforms;
	cl_uint					numDevices;
	cl_device_id			device, *pdevice;
	cl_int	    status;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_int ret;
    float mem[MEM_SIZE];

    FILE *fp;
    const char fileName[] = "./opencl/online.cl";
    size_t source_size;
    char *source_str;
    cl_int i;




	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != 0) {
		printf("clGetPlatformIDs status = %d", status);
	}
	platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	if (platforms == NULL)
		return false;

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != 0) {
		printf("clGetPlatformIDs status = %d", status);
	}

	platform = platforms[1];
	free(platforms);

	//Query the platform and choose the first GPU device if has one.
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (status != 0) {
		printf("clGetDeviceIDs status = %d", status);
	}
	pdevice = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
	if (pdevice == NULL)
		return false;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, pdevice, NULL);
	if (status != 0) {
		printf("clGetDeviceIDs status = %d", status);
	}
	device = pdevice[0];
	free(pdevice);
	printDevInfo(platform,device);

    /* Load kernel source code */
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
    fclose( fp );

    /*Initialize Data */
    for( i = 0; i < MEM_SIZE; i++ ) {
        mem[i] = i;
    }

    /* Create OpenCL Context */
    context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device, 0, &ret);

    /* Create memory buffer*/
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);

    /* Transfer data to memory buffer */
    ret = clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);

    /* Create Kernel program from the read in source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "vecAdd", &ret);

    /* Set OpenCL kernel argument */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);

	size_t global_work_size[3] = {MEM_SIZE, 0, 0};
    size_t local_work_size[3]  = {MEM_SIZE, 0, 0};

    /* Execute OpenCL kernel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, 0, global_work_size, local_work_size, 0, NULL, NULL);

    /* Transfer result from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);

    /* Display result */
    for(i=0; i<MEM_SIZE; i++) {
        printf("mem[%d] : %f\n", i, mem[i]);
    }

    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(source_str);
	char c = getchar();
    return 0;
}

 
static void strInfo(cl_device_id   dev, cl_device_info  infoItem, const char * name)
{
	void        *pValue = NULL;
	size_t      valueSize;

	clGetDeviceInfo(dev, infoItem, NULL, NULL, &valueSize);
	pValue = (char*)malloc(valueSize * sizeof(char));
	clGetDeviceInfo(dev, infoItem, valueSize, pValue, NULL);
	printf("opencl %s: %s\n",name, pValue);
	free(pValue);
}
static void valInfo(cl_device_id   dev, cl_device_info  infoItem, const char * name)
{
	cl_ulong     pValue = NULL;

	clGetDeviceInfo(dev, infoItem, sizeof(pValue), &pValue, NULL);
	printf("opencl %s: %d\n", name, pValue);
}
int printDevInfo(cl_platform_id platform, cl_device_id device)
{	 

	//long for usual
	valInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, "max computeUnit");
	valInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, "max paraSize");
	valInfo(device, CL_DEVICE_MAX_SAMPLERS, "max sample");

	//string for usual
	strInfo(device, CL_DEVICE_NAME, "device name");
	strInfo(device, CL_DEVICE_VENDOR, "vendor");
	strInfo(device, CL_DEVICE_EXTENSIONS, "extend");
	strInfo(device, CL_DEVICE_PROFILE, "profile"); 
	strInfo(device, CL_DEVICE_VERSION, "ver");
	return 0;
}
