#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void vecAdd(__global float* a)
{
    int gid = get_global_id(0);

	printf("gid=%d,a[gid]=%f\n", gid,a[gid]);
    a[gid] += a[gid];
	
}
