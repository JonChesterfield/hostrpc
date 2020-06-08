#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_HPP_INCLUDED

#include "common.h"

namespace hostrpc
{
namespace config
{
struct fill
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(page, dv, sizeof(hostrpc::page_t));
  };
};

struct use
{
  void operator()(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(dv, page, sizeof(hostrpc::page_t));
  };
};

struct operate
{
  void operator()(hostrpc::page_t *page, void *)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t &line = page->cacheline[c];
        std::swap(line.element[0], line.element[7]);
        std::swap(line.element[1], line.element[6]);
        std::swap(line.element[2], line.element[5]);
        std::swap(line.element[3], line.element[4]);
        for (unsigned i = 0; i < 8; i++)
          {
            line.element[i]++;
          }
      }
  }
};

// 128 won't suffice, probably need the whole structure to be templated
using x64_amdgcn_client =
    hostrpc::client<128, hostrpc::copy_functor_memcpy_pull, fill, use,
                    hostrpc::nop_stepper>;

using x64_amdgcn_server =
    hostrpc::server<128, hostrpc::copy_functor_memcpy_pull, operate,
                    hostrpc::nop_stepper>;

}  // namespace config

// need to allocate buffers for both together
// one needs to reside on the host, one on the gpu
struct x64_amdgcn_pair
{
  x64_amdgcn_pair();
  ~x64_amdgcn_pair();
  config::x64_amdgcn_client *client;
  config::x64_amdgcn_server *server;
  void *state;
};

}  // namespace hostrpc

#endif
