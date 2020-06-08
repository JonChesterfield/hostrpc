#include "x64_host_amdgcn_client.hpp"
// #include "catch.hpp"
#include "hsa.hpp" // hsa includes stdlib

#include <string>

namespace hostrpc
{
x64_amdgcn_pair::x64_amdgcn_pair() {}
~x64_amdgcn_pair::x64_amdgcn_pair() {}
}  // namespace hostrpc

namespace
{
hsa::init global_state;
}

void function wip()
  std::vector<hsa_agent_t> kernel_agents;
  std::vector<hsa_agent_t> other_agents;
  hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto features = hsa::agent_get_info_feature(agent);
    std::vector<hsa_agent_t>* list =
        (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) ? &kernel_agents
                                                       : &other_agents;
    list->push_back(agent);
    return HSA_STATUS_SUCCESS;
  });

  printf("Found %zu kernel agents\n", kernel_agents.size());
assert(kernel_agents.size() > 0);
  printf("Found %zu other agents\n", other_agents.size());

  hsa_region_t fine = region_fine_grained(kernel_agents[0]);


  constexpr size_t N = 128;

  page_t * client_buffer = reinterpret_cast<page_t*>(hsa::allocate(fine, N * sizeof(page_t)));
  page_t * server_buffer = reinterpret_cast<page_t*>(hsa::allocate(fine, N * sizeof(page_t)));

  hostrpc::copy_functor_memcpy_pull cp;
  hostrpc::nop_stepper st;

  using mt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
                         hsa_allocte_slot_bitmap_data>;
  using mailbox_ptr_t = std::unique_ptr<mt::slot_bitmap_data_t,
                                        mt::slot_bitmap_data_t::deleter>;

    using lt = slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE,
                         hsa_allocte_slot_bitmap_data>;
  using lockarray_ptr_t = std::unique_ptr<lt::slot_bitmap_data_t,
                                          lt::slot_bitmap_data_t::deleter>;

    

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
