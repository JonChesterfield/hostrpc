#include <stdio.h>

#include "allocator.hpp"
#include "client_server_pair.hpp"
#include "openmp_plugins.hpp"

namespace hostrpc
{
template <typename SZ, int device_num>
using x64_device_type_base =
    client_server_pair_t<SZ, uint64_t,
                         allocator::openmp_shared<alignof(page_t)>,
                         allocator::openmp_shared<64>, allocator::host_libc<64>,
                         allocator::openmp_device<64, device_num>>;

template <typename SZ, int device_num>
struct x64_device_type : public x64_device_type_base<SZ, device_num>
{
  using base = x64_device_type_base<SZ, device_num>;
  HOSTRPC_ANNOTATE x64_device_type(SZ sz)
      : base(sz, typename base::AllocBuffer(),
             typename base::AllocInboxOutbox(), typename base::AllocLocal(),
             typename base::AllocRemote())
  {
  }
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
