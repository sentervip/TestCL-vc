// Convolution.cl
//
//    This is a simple kernel performing convolution.

__kernel void convolve(
	const __global  uint * const input,
    __constant uint * const mask,
    __global  uint * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

	
    uint sum = 0;
    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;

        for (int c = 0; c < maskWidth; c++)
        {
			sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
	printf("x=%d,y=%d\n", x,y);
}