#ifndef X64_HOST_PTX_CLIENT_HPP_INCLUDED
#define X64_HOST_PTX_CLIENT_HPP_INCLUDED

#include "detail/client_impl.hpp"
#include "detail/common.hpp"
#include "detail/server_impl.hpp"

#include "memory.hpp"
#include "test_common.hpp"


#if defined(__x86_64__)
#include <cstdio>
#include <cstdlib>
#endif

namespace hostrpc
{

  namespace cuda
  {
    void * allocate_gpu(size_t align, size_t size);
    void deallocate_gpu(void*);

    void * allocate_shared(size_t align, size_t size);
    void deallocate_shared(void*);

    void * device_ptr_from_host_ptr(void*);
  }

#if defined(__x86_64__)
  template <typename T>
struct x64_ptx_pair
{
  x64_ptx_pair(size_t align,  size_t element_count) :x64(nullptr), ptx(nullptr)
  {
    size_t element_size = sizeof(T);
    size_t bytes = element_size * element_count;
    void * void_x64_buffer = cuda::allocate_shared(align, bytes);
    if (void_x64_buffer) {
      void * void_ptx_buffer = cuda::device_ptr_from_host_ptr(void_x64_buffer);
      if (void_ptx_buffer) {
        x64 = hostrpc::careful_array_cast<T>(void_x64_buffer, element_count);
        assert(x64);
        ptx = hostrpc::careful_array_cast<T>(void_ptx_buffer, element_count);
        assert(ptx);
        return;
      }
    }

    fprintf(stderr, "Failed to construct x64_ptx_pair_T, exit\n");
    exit(1);
  }
  T * x64 ;
  T * ptx ;
};
#endif
  
template <typename SZ, typename Fill, typename Use, typename Operate,
          typename Clear>
struct x64_ptx_pair_T
{
  using Copy = copy_functor_given_alias;
  using Step = nop_stepper;

  using Word = uint32_t;
  using client_type =
      client_impl<Word, SZ, Copy, Fill, Use, Step, counters::client>;
  using server_type =
      server_impl<Word, SZ, Copy, Operate, Clear, Step, counters::server>;

  client_type client;
  server_type server;

  x64_ptx_pair_T(SZ sz)
  {
#if defined(__x86_64__)
    size_t N = sz.N();

    x64_ptx_pair<page_t> buffer (alignof(page_t), N);

    // area read/write by either
    x64_ptx_pair<typename server_type::outbox_t::Ty> send(64, N);
    x64_ptx_pair<typename server_type::inbox_t::Ty> recv(64, N);

    // only accessed by client
    // ....


        // only accessed by server
    auto server_active =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::lock_t>(N);

    auto server_staging =
        x64_allocate_slot_bitmap_data_alloc<typename server_type::staging_t>(N);

      

#if 0
    client = {sz,           client_active,  recv,
              send,         client_staging, server_buffer,
              client_buffer};

    server = {sz,           server_active,  send,
              recv,         server_staging, client_buffer,
              server_buffer};
#endif
    assert(client.size() == N);
    assert(server.size() == N);
    
    return;


#else
    (void)sz;
#endif
  }

  
};
}

#endif
