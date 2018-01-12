
__kernel void local_test(__local char *p, int local_size) {
    for (int i=0; i<local_size; i++) {
        p[i] = i;               
    }
	printf("localSize=%d\n",local_size);
}
