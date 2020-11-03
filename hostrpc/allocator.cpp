#include "allocator.hpp"

void instantiate_hsa()
{
  uint64_t handle = 42;
  hostrpc::allocator::hsa<64> instance(handle);

  auto a = instance.allocate(4);
  (void)a.local();
  (void)a.remote();
  (void)a.destroy();
}

void instantiate_libc()
{
  hostrpc::allocator::host_libc<64> instance;
  auto a = instance.allocate(8);
  (void)a.local();
  (void)a.remote();
  (void)a.destroy();
}

void instantiate_cuda_shared()
{
  hostrpc::allocator::cuda_shared<128> instance;
  auto a = instance.allocate(16);
  (void)a.local();
  (void)a.remote();
  (void)a.destroy();
}

void instantiate_cuda_gpu()
{
  hostrpc::allocator::cuda_gpu<256> instance;
  auto a = instance.allocate(32);
  (void)a.local();
  (void)a.remote();
  (void)a.destroy();
}

int main()
{
  instantiate_hsa();
  instantiate_libc();
  instantiate_cuda_shared();
  instantiate_cuda_gpu();
}
