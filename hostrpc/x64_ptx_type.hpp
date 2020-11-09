#ifndef X64_NVPTX_TYPE_HPP_INCLUDED
#define X64_NVPTX_TYPE_HPP_INCLUDED

#include "base_types.hpp"
#include "detail/client_impl.hpp"
#include "detail/platform_detect.h"
#include "detail/server_impl.hpp"

#include "allocator_cuda.hpp"
#include "allocator_libc.hpp"
#include "host_client.hpp"
#include "memory.hpp"

#if HOSTRPC_HOST
#include <stdio.h>
#include <stdlib.h>
#endif

namespace hostrpc
{
struct x64_ptx_type
{
  using SZ = hostrpc::size_runtime;
  using Copy = copy_functor_given_alias;
  using Word = uint32_t;
  using client_type = client_impl<Word, SZ, Copy, counters::client_nop>;
  using server_type = server_impl<Word, SZ, Copy, counters::server_nop>;

  client_type client;
  server_type server;

  using AllocBuffer = hostrpc::allocator::cuda_shared<alignof(page_t)>;
  using AllocInboxOutbox = hostrpc::allocator::cuda_shared<64>;

  using AllocLocal = hostrpc::allocator::host_libc<64>;
  using AllocRemote = hostrpc::allocator::cuda_gpu<64>;

  using storage_type = allocator::store_impl<AllocBuffer, AllocInboxOutbox,
                                             AllocLocal, AllocRemote>;

  storage_type storage;
  x64_ptx_type(size_t N)
  {
    N = hostrpc::round64(N);
    AllocBuffer alloc_buffer;
    AllocInboxOutbox alloc_inbox_outbox;
    AllocLocal alloc_local;
    AllocRemote alloc_remote;

    SZ sz(N);
    storage = host_client(alloc_buffer, alloc_inbox_outbox, alloc_local,
                          alloc_remote, sz, &server, &client);
    if (!storage.valid())
      {
#if HOSTRPC_HOST
        fprintf(stderr, "x64_ptx_type construction failed\n");
        exit(1);
#endif
      }
    else
      {
        storage.dump();
      }
  }

  ~x64_ptx_type() { storage.destroy(); }
  x64_ptx_type(const x64_ptx_type &) = delete;
};

}  // namespace hostrpc

#endif
