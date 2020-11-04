#include "allocator_cuda.hpp"
#include "allocator_hsa.hpp"
#include "allocator_libc.hpp"

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

  auto s = hostrpc::allocator::cuda_shared<128>{}.allocate(16);
  auto g = hostrpc::allocator::cuda_gpu<256>{}.allocate(48);

  hostrpc::allocator::raw_store().destroy();
  raw_store(s);
  raw_store(s, s);
  raw_store(s, s, g, s);

  auto st = raw_store(s, g);
  auto r = st.destroy();
  (void)r;
}
