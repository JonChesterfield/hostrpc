#include "x64_host_amdgcn_client.hpp"

namespace hostrpc
{
x64_amdgcn_pair::x64_amdgcn_pair() {}
x64_amdgcn_pair::~x64_amdgcn_pair() {}
}  // namespace hostrpc

static void* alloc_from_region(hsa_region_t region, size_t size)
{
  void* res;
  hsa_status_t r = hsa_memory_allocate(region, size, &res);
  if (r == HSA_STATUS_SUCCESS)
    {
      return res;
    }
  else
    {
      return nullptr;
    }
}

void wip(hsa_region_t fine)
{
  using namespace hostrpc;
  constexpr size_t N = 128;

  hostrpc::page_t* client_buffer =
      reinterpret_cast<page_t*>(alloc_from_region(fine, N * sizeof(page_t)));
  hostrpc::page_t* server_buffer =
      reinterpret_cast<page_t*>(alloc_from_region(fine, N * sizeof(page_t)));

  hostrpc::copy_functor_memcpy_pull cp;
  hostrpc::nop_stepper st;

  using mt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                         hsa_allocate_slot_bitmap_data>;

  using lt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE,
                         hsa_allocate_slot_bitmap_data>;

  mt::slot_bitmap_data_t* send_data = mt::slot_bitmap_data_t::alloc(fine);
  mt::slot_bitmap_data_t* recv_data = mt::slot_bitmap_data_t::alloc(fine);

  lt::slot_bitmap_data_t* client_active_data =
      lt::slot_bitmap_data_t::alloc(fine);

  lt::slot_bitmap_data_t* server_active_data =
      lt::slot_bitmap_data_t::alloc(fine);

  mt send(send_data);
  mt recv(recv_data);
  lt client_active(client_active_data);
  lt server_active(server_active_data);

  hostrpc::config::x64_amdgcn_client client(cp, recv, send, client_active,
                                            server_buffer, client_buffer, st,
                                            config::fill{}, config::use{});

  hostrpc::config::x64_amdgcn_server server(cp, send, recv, server_active,
                                            client_buffer, server_buffer, st,
                                            config::operate{});

  (void)client;
  (void)server;

  mt::slot_bitmap_data_t::free(send_data);
  mt::slot_bitmap_data_t::free(recv_data);

  lt::slot_bitmap_data_t::free(client_active_data);
  lt::slot_bitmap_data_t::free(server_active_data);
}
