/* System includes */
#include <stdio.h>
#include <stdlib.h>

/* OpenCL includes */
#include <CL/cl.h>

void OpenCLInit(cl_platform_id *clPlatform, cl_device_id *clDevice, cl_context *clContext)
{
	cl_uint numPlatforms = 0;   //GPU计算平台个数  
	cl_platform_id platform = NULL;
	clGetPlatformIDs(0, NULL, &numPlatforms);

	//获得平台列表  
	cl_platform_id * platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	//轮询各个opencl设备  
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		char pBuf[100];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(pBuf), pBuf, NULL);
		printf("getPlat[%d]: %s\n",i, pBuf);
		platform = platforms[i];
		//break;  
	}

	*clPlatform = platform;

	free(platforms);

	cl_int status = 0;

	//获得GPU设备  
	cl_device_id device;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	*clDevice = device;

	//生成上下文  
	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &status);
	*clContext = context;
}
static void strInfo(cl_device_id   dev, cl_device_info  infoItem, const char * name)
{
	void        *pValue = NULL;
	size_t      valueSize;

	clGetDeviceInfo(dev, infoItem, NULL, NULL, &valueSize);
	pValue = (char*)malloc(valueSize * sizeof(char));
	clGetDeviceInfo(dev, infoItem, valueSize, pValue, NULL);
	printf("opencl %s: %s\n", name, pValue);
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

int BuildKernel(cl_device_id* device, cl_context* context, cl_program *program, const char* fileName)
{
	cl_int errNum;
	//char *pSrcStr = NULL;
	FILE *fp;
	char *outFileData;
	long fileSize;

	fp = fopen(fileName, "r");
	if (!fp) {
		printf("Could not open file: %s\n", fileName);
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_END)) {
		printf("Error reading the file\n");
		exit(-1);
	}
	fileSize = ftell(fp);
	if (fileSize < 0) {
		printf("Error reading the file\n");
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_SET)) {
		printf("Error reading the file\n");
		exit(-1);
	}

	// Read the contents
	outFileData = (char*)malloc(fileSize + 1);
	if (!outFileData) {
		exit(-1);
	}
	if (fread(outFileData, fileSize, 1, fp) != 1) {
		printf("Error reading the file\n");
		exit(-1);
	}

	outFileData[fileSize] = '\0';
	if (fclose(fp)) {
		printf("Error closing the file\n");
		exit(-1);
	}

	//readFile2(fileName, &pSrcStr);
	*program = clCreateProgramWithSource(*context, 1,
		(const char**)&outFileData,
		NULL, NULL);
	if (program == NULL)
	{
		printf("Failed to create CL program from source\n");  
		exit(-1);
	}

	errNum = clBuildProgram(*program, 1, device, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		printf("Error in kernel: %s\n",buildLog);
		clReleaseProgram(*program);
		exit(-1);
	}
	free(outFileData);
	return 0;
}

void check(cl_int status) {

   if (status != CL_SUCCESS) {
      printf("OpenCL error (%d)\n", status);
	  char c = getchar();
      exit(-1);
   }
}
char* readFile(const char *filename)
{

	FILE *fp;
	char *fileData;
	long fileSize;

	/* Open the file */
	fp = fopen(filename, "r");
	if (!fp) {
		printf("Could not open file: %s\n", filename);
		exit(-1);
	}

	/* Determine the file size */
	if (fseek(fp, 0, SEEK_END)) {
		printf("Error reading the file\n");
		exit(-1);
	}
	fileSize = ftell(fp);
	if (fileSize < 0) {
		printf("Error reading the file\n");
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_SET)) {
		printf("Error reading the file\n");
		exit(-1);
	}

	/* Read the contents */
	fileData = (char*)malloc(fileSize + 1);
	if (!fileData) {
		exit(-1);
	}
	if (fread(fileData, fileSize, 1, fp) != 1) {
		printf("Error reading the file\n");
		exit(-1);
	}

	/* Terminate the string */
	fileData[fileSize] = '\0';

	/* Close the file */
	if (fclose(fp)) {
		printf("Error closing the file\n");
		exit(-1);
	}

	return fileData;
}
int readFile2(const char *filename, char * outFileData) 
{
	FILE *fp;
	char *fileData;
	long fileSize;

	/* Open the file */
	fp = fopen(filename, "r");
	if (!fp) {
		printf("Could not open file: %s\n", filename);
		exit(-1);
	}

	/* Determine the file size */
	if (fseek(fp, 0, SEEK_END)) {
		printf("Error reading the file\n");
		exit(-1);
	}
	fileSize = ftell(fp);
	if (fileSize < 0) {
		printf("Error reading the file\n");
		exit(-1);
	}
	if (fseek(fp, 0, SEEK_SET)) {
		printf("Error reading the file\n");
		exit(-1);
	}

	/* Read the contents */
	outFileData = (char*)malloc(fileSize + 1);
	if (!outFileData) {
		exit(-1);
	}
	if (fread(outFileData, fileSize, 1, fp) != 1) {
		printf("Error reading the file\n");
		exit(-1);
	}

	/* Terminate the string */
	outFileData[fileSize] = '\0';

	/* Close the file */
	if (fclose(fp)) {
		printf("Error closing the file\n");
		exit(-1);
	}
	//outFileData = fileData;
	return 0;
}

void printCompilerError(cl_program program, cl_device_id device) {
   cl_int status;

   size_t logSize;
   char *log;

   /* Get the log size */
   status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
               0, NULL, &logSize);
   check(status);

   /* Allocate space for the log */
   log = (char*)malloc(logSize);
   if (!log) {
      exit(-1);
   }

   /* Read the log */
   status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
               logSize, log, NULL);
   check(status);

   /* Print the log */
   printf("%s\n", log);
}


