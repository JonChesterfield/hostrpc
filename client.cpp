#include "client.hpp"
#define N 128

struct copy_functor_nop
    : public hostrpc::copy_functor_interface<copy_functor_nop>
{
};

using client_type =
    hostrpc::client<N, hostrpc::x64_x64_bitmap_types, copy_functor_nop,
                    hostrpc::fill_nop, hostrpc::use_nop, hostrpc::nop_stepper>;

extern "C" __attribute__((noinline)) void client_instance_direct(client_type& c)
{
  for (;;)
    {
      c.rpc_invoke<true>(nullptr);
      c.rpc_invoke<false>(nullptr);
    }
}

extern "C" __attribute__((noinline)) void client_instance_from_pointers(
    const hostrpc::mailbox_t<N> inbox, hostrpc::mailbox_t<N> outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  client_type c = {inbox, outbox, active, remote_buffer, local_buffer};
  client_instance_direct(c);
}
