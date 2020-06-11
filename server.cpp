#include "server.hpp"
#include "memory.hpp"

#define N 128
void server_instance(
    const hostrpc::mailbox_t<N> inbox, hostrpc::mailbox_t<N> outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  struct copy_functor_nop
      : public hostrpc::copy_functor_interface<copy_functor_nop>
  {
  };

  using server_type = hostrpc::server<N, copy_functor_nop, hostrpc::operate_nop,
                                      hostrpc::nop_stepper>;

  server_type s = {inbox, outbox, active, remote_buffer, local_buffer};

  for (;;)
    {
      s.rpc_handle(nullptr);
    }
}
