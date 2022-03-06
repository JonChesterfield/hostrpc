#include <stdio.h>

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "openmp_plugins.hpp"

namespace hostrpc
{
template <typename SZ, int device_num>
using x64_device_type_base =
    client_server_pair_t<SZ, arch::x64, arch::openmp_target<device_num>>;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz) : base(sz, {}, {}) {}
};
}  // namespace hostrpc

int main()
{
  using SZ = hostrpc::size_compiletime<1920>;
  constexpr static int device_num = 0;

  SZ sz;

  {
    hostrpc::x64_device_type<SZ, device_num> p(sz);
  }

  {
    hostrpc::x64_device_type<SZ, device_num> p(sz);
    p.storage.destroy();
  }

  return 0;
}
