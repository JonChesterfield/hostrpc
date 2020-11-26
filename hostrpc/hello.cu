// https://forums.developer.nvidia.com/t/does-anybody-have-experience-on-cudahostregister-zero-copy-memory/22539/3

#include <stdio.h>

#define SIZE 10

#include <cuda.h>

// Kernel definition, see also section 4.2.3 of Nvidia Cuda Programming Guide

__global__ void vecAdd(float *A, float *B, float *C)
{
  // threadIdx.x is a built-in variable provided by CUDA at runtime

  int i = threadIdx.x;

  //	A[i] = 0;
  //	B[i] = i;
  C[i] = A[i] + B[i];
  printf("Kernel: A[%d]=%f, B[%d]=%f, C[%d]=%f\n", i, A[i], i, B[i], i, C[i]);
}

int main()
{
  int N = SIZE;

  //	round up the size of the array to be a multiple of the page size

  size_t memsize = ((SIZE * sizeof(float) + 4095) / 4096) * 4096;

  cudaDeviceProp deviceProp;

  // Get properties and verify device 0 supports mapped memory

  cudaGetDeviceProperties(&deviceProp, 0);

  if (!deviceProp.canMapHostMemory)
    {
      fprintf(stderr, "Device %d cannot map host memory!\n", 0);

      exit(EXIT_FAILURE);
    }

  // set the device flags for mapping host memory

  cudaSetDeviceFlags(cudaDeviceMapHost);

  float *A, *B, *C;

  float *devPtrA, *devPtrB, *devPtrC;

  //	use valloc instead of malloc

  A = (float *)valloc(memsize);

  B = (float *)valloc(memsize);

  C = (float *)valloc(memsize);

  cudaHostRegister(A, memsize, cudaHostRegisterMapped);

  cudaHostRegister(B, memsize, cudaHostRegisterMapped);

  cudaHostRegister(C, memsize, cudaHostRegisterMapped);

  for (int i = 0; i < SIZE; i++)
    {
      A[i] = B[i] = i;
    }

  cudaHostGetDevicePointer((void **)&devPtrA, (void *)A, 0);
  cudaHostGetDevicePointer((void **)&devPtrB, (void *)B, 0);
  cudaHostGetDevicePointer((void **)&devPtrC, (void *)C, 0);

  vecAdd<<<1, N>>>(devPtrA, devPtrB, devPtrC);

  cudaDeviceSynchronize();

  for (int i = 0; i < SIZE; i++) printf("C[%d]=%f\n", i, C[i]);

  cudaHostUnregister(A);
  cudaHostUnregister(B);
  cudaHostUnregister(C);

  free(A);
  free(B);
  free(C);
}
