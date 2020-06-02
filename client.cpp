#include "client.hpp"
#define N 128

void client_instance(
    const hostrpc::mailbox_t<N>* inbox, hostrpc::mailbox_t<N>* outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  hostrpc::nop_stepper step;
  auto fill = hostrpc::fill_nop;
  auto use = hostrpc::use_nop;
  hostrpc::copy_functor_x64_x64 cp;
  auto s = hostrpc::make_client(cp, inbox, outbox, active, remote_buffer,
                                local_buffer, step, fill, use);

  for (;;)
    {
      s.rpc_invoke<true>();
      s.rpc_invoke<false>();
    }
}

extern "C" void instantiate_try_garbage_collect_word_client(
    const hostrpc::mailbox_t<N>* inbox, hostrpc::mailbox_t<N>* outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE>* active, uint64_t w)
{
  auto c = [](uint64_t i, uint64_t) -> uint64_t { return i; };
  hostrpc::try_garbage_collect_word<N, decltype(c)>(c, inbox, outbox, active,
                                                    w);
}
