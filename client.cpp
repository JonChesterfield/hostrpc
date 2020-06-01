#include "client.hpp"
#define N 128

void client_instance(
    const hostrpc::mailbox_t<N>* inbox, hostrpc::mailbox_t<N>* outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
    hostrpc::page_t* buffer)
{
  hostrpc::nop_stepper step;
  auto fill = hostrpc::fill_nop;
  auto use = hostrpc::use_nop;

  auto s = hostrpc::client<N, decltype(fill), decltype(use), decltype(step)>{
      inbox, outbox, active, buffer, step, fill, use};

  for (;;)
    {
      s.rpc_invoke<true>();
      s.rpc_invoke<false>();
    }
}
