#include "allocator.hpp"

int main()
{
  uint64_t handle = 42;
  hostrpc::allocator::hsa_ex instance(handle);

  auto a = instance.allocate(4, 4);
  auto l = a.local();
  auto r = a.remote();

  int rc = a.destroy();
  (void)rc;
  (void)l;
  (void)r;
}
