#ifndef HOSTRPC_X64_HOST_X64_CLIENT_HPP_INCLUDED
#define HOSTRPC_X64_HOST_X64_CLIENT_HPP_INCLUDED

#include "client.hpp"
#include "server.hpp"

namespace hostrpc
{
namespace x64_host_x64_client
{
struct fill
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(page, dv, sizeof(hostrpc::page_t));
  };
};

struct use
{
  static void call(hostrpc::page_t *page, void *dv)
  {
    __builtin_memcpy(dv, page, sizeof(hostrpc::page_t));
  };
};

struct operate
{
  static void call(hostrpc::page_t *page, void *)
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

}  // namespace x64_host_x64_client

template <size_t N>
using x64_x64_client =
    hostrpc::client<N, hostrpc::copy_functor_memcpy_pull,
                    hostrpc::x64_host_x64_client::fill,
                    hostrpc::x64_host_x64_client::use, hostrpc::nop_stepper>;

template <size_t N>
using x64_x64_server = hostrpc::server<N, hostrpc::copy_functor_memcpy_pull,
                                       hostrpc::x64_host_x64_client::operate,
                                       hostrpc::nop_stepper>;

template <size_t N>
struct x64_x64_pair
{
  x64_x64_client<N> client;
  x64_x64_server<N> server;

  hostrpc::page_t client_buffer[N];
  hostrpc::page_t server_buffer[N];
  x64_x64_pair()
  {
    using namespace hostrpc;
#if 0
    size_t buffer_size = sizeof(page_t) * N;
      page_t *client_buffer =
    new (buffer_size, aligned_alloc(alignof(page_t), buffer_size)) page_t[N];
    page_t *server_buffer =
      new (buffer_size, aligned_alloc(alignof(page_t), buffer_size) page_t[N];
#endif
    assert(client_buffer != server_buffer);

    slot_bitmap_data<N> *send_data = x64_allocate_slot_bitmap_data<N>();
    slot_bitmap_data<N> *recv_data = x64_allocate_slot_bitmap_data<N>();
    slot_bitmap_data<N> *client_locks_data = x64_allocate_slot_bitmap_data<N>();
    slot_bitmap_data<N> *server_locks_data = x64_allocate_slot_bitmap_data<N>();

    slot_bitmap_all_svm<N> send(send_data);
    slot_bitmap_all_svm<N> recv(recv_data);
    slot_bitmap_device<N> client_locks(client_locks_data);
    slot_bitmap_device<N> server_locks(server_locks_data);

    client = {recv, send, client_locks, server_buffer, client_buffer};
    server = {send, recv, server_locks, client_buffer, server_buffer};
  }
  ~x64_x64_pair()
  {
    assert(client.inbox.data() == server.outbox.data());
    assert(client.outbox.data() == server.inbox.data());

    hostrpc::x64_allocate_slot_bitmap_data_deleter<N> del;
    del(client.inbox.data());
    del(server.inbox.data());
    del(client.active.data());
    del(server.active.data());

    assert(client.local_buffer != server.local_buffer);
    // free(client.local_buffer);
    // free(server.local_buffer);
  }
};

}  // namespace hostrpc

#endif
