// https://forums.developer.nvidia.com/t/does-anybody-have-experience-on-cudahostregister-zero-copy-memory/22539/3

#include <stdio.h>
#include <sys/mman.h>

#define SIZE 10

#include <cuda.h>

// Kernel definition, see also section 4.2.3 of Nvidia Cuda Programming Guide

__global__ void vecAdd(float *A, float *B, float *C)
{
  int i = threadIdx.x;

  //	A[i] = 0;
  //	B[i] = i;
  C[i] = A[i] + B[i];
  printf("Kernel: A[%d]=%f, B[%d]=%f, C[%d]=%f\n", i, A[i], i, B[i], i, C[i]);
}

void *map_alloc(size_t size)
{
  return mmap(NULL, size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
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

  fprintf(stderr, "uni addr: %u\n", deviceProp.unifiedAddressing);
  fprintf(stderr, "can use host pointer: %u\n",
          deviceProp.canUseHostPointerForRegisteredMem);

  // set the device flags for mapping host memory

  cudaSetDeviceFlags(cudaDeviceMapHost);

  float *A, *B, *C;

  float *devPtrA, *devPtrB, *devPtrC;

  //	use valloc instead of malloc
  A = (float *)map_alloc(memsize);
  B = (float *)map_alloc(memsize);
  C = (float *)map_alloc(memsize);

  cudaHostRegister(A, memsize, cudaHostRegisterMapped);
  cudaHostRegister(B, memsize, cudaHostRegisterMapped);
  cudaHostRegister(C, memsize, cudaHostRegisterMapped);

  for (int i = 0; i < SIZE; i++)
    {
      A[i] = B[i] = i;
    }

  cudaHostGetDevicePointer((void **)&devPtrA, (void *)A, 0);
  fprintf(stderr, "%p =? %p\n", devPtrA, A);

  {
    cudaPointerAttributes attr;
    cudaError_t rc = cudaPointerGetAttributes(&attr, (void *)A);
    if (rc != cudaSuccess)
      {
        fprintf(stderr, "fail\n");
      }
    fprintf(stderr, "prop[%p]: dev %u, dptr %p, hptr %p\n", A, attr.device,
            attr.devicePointer, attr.hostPointer);
  }

  {
    cudaPointerAttributes attr;
    cudaError_t rc = cudaPointerGetAttributes(&attr, (void *)devPtrA);
    if (rc != cudaSuccess)
      {
        fprintf(stderr, "fail\n");
      }
    fprintf(stderr, "prop[%p]: dev %u, dptr %p, hptr %p\n", devPtrA,
            attr.device, attr.devicePointer, attr.hostPointer);
  }

  cudaHostGetDevicePointer((void **)&devPtrB, (void *)B, 0);
  cudaHostGetDevicePointer((void **)&devPtrC, (void *)C, 0);

  vecAdd<<<1, N>>>(devPtrA, devPtrB, devPtrC);

  cudaDeviceSynchronize();

  for (int i = 0; i < SIZE; i++) printf("C[%d]=%f\n", i, C[i]);

  cudaHostUnregister(A);
  cudaHostUnregister(B);
  cudaHostUnregister(C);

  // free(A);
  munmap(A, memsize);
  munmap(B, memsize);
  munmap(C, memsize);
}
