#include "allocator.hpp"

int main()
{
  hostrpc::allocator::hsa_ex instance;

  auto a = instance.allocate(4, 4);
  auto l = a.local();
  auto r = a.remote();

  int rc = a.destroy();
  (void)rc;
  (void)l;
  (void)r;
}
