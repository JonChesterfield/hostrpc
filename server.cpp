#include "server.hpp"
#define N 128
void server_instance(
    const hostrpc::mailbox_t<N>* inbox, hostrpc::mailbox_t<N>* outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
    const hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  hostrpc::nop_stepper step;
  auto operate = hostrpc::operate_nop;

  auto s = hostrpc::make_server(inbox, outbox, active, remote_buffer,
                                local_buffer, step, operate);

  for (;;)
    {
      s.rpc_handle();
    }
}
