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
  using mailbox_ptr_t =
      std::unique_ptr<mt::slot_bitmap_data_t, mt::slot_bitmap_data_t::deleter>;

  using lt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE,
                         hsa_allocate_slot_bitmap_data>;
  using lockarray_ptr_t =
      std::unique_ptr<lt::slot_bitmap_data_t, lt::slot_bitmap_data_t::deleter>;

  mailbox_ptr_t send_data(mailbox_t<N>::slot_bitmap_data_t::alloc(fine));
  mailbox_ptr_t recv_data(mailbox_t<N>::slot_bitmap_data_t::alloc(fine));
  lockarray_ptr_t client_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc(fine));
  lockarray_ptr_t server_active_data(
      lockarray_t<N>::slot_bitmap_data_t::alloc(fine));

  mailbox_t<N> send(send_data.get());
  mailbox_t<N> recv(recv_data.get());
  lockarray_t<N> client_active(client_active_data.get());
  lockarray_t<N> server_active(server_active_data.get());

  x64_amdgcn_client client(cp, recv, send, client_active, server_buffer,
                           client_buffer, st, config::fill{}, config::use{});

  (void)client;
}
