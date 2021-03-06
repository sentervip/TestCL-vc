__kernel
void convolution(
	__global          float* inputImage,
	__global          float* outputImage,
	__global          float* filter,
	__global          int filterWidth)
 //               sampler_t sampler)
{
   /* Store each work-item’s unique row and column */
   int column = get_global_id(0);
   int row = get_global_id(1);
   
   /* Half the width of the filter is needed for indexing
    * memory later */
   int halfWidth = (int)(filterWidth/2);
   
   /* All accesses to images return data as four-element vector
    * (i.e., float4), although only the ’x’ component will contain
    * meaningful data in this code */
   int4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
   
   /* Iterator for the filter */
   int filterIdx = 0;
   
   /* Each work-item iterates around its local area based on the
    * size of the filter */
   int2 coords; // Coordinates for accessing the image
   
   /* Iterate the filter rows */
   for(int i = -halfWidth; i <= halfWidth; i++) 
   {
      coords.y = row + i;
      /* Iterate over the filter columns */
      for(int j = -halfWidth; j <= halfWidth; j++) 
      {
         coords.x = column + j;
         
         /* Read a pixel from the image. A single channel image
          * stores the pixel in the ’x’ coordinate of the returned
          * vector. */
            int4 pixel;         
        // pixel = read_imagef(inputImage, sampler, coords);
			pixel = inputImage[coords.x];
            sum.x += pixel.x * filter[filterIdx++];
      }
   }
   
   /* Copy the data to the output image */
   coords.x = column;
   coords.y = row;
   //write_imagef(outputImage, coords, sum);
   outputImage[coords.x+coords.y*4] = sum[coords.x+ coords.y * 4];
}
