#include "server.hpp"
#define N 128
void server_instance(
    const hostrpc::mailbox_t<N>* inbox, hostrpc::mailbox_t<N>* outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
    hostrpc::page_t* buffer)

{
  hostrpc::nop_stepper step;
  auto operate = hostrpc::operate_nop;

  auto s = hostrpc::server<N, decltype(operate), hostrpc::nop_stepper>{inbox,  outbox, active,
                                                    buffer, step,   operate};

  for (;;)
    {
      s.rpc_handle();
    }
}
