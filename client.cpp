#include "client.hpp"
#define N 128

void client_instance(
    const hostrpc::mailbox_t<N> inbox, hostrpc::mailbox_t<N> outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  hostrpc::copy_functor_memcpy_pull cp;

  using client_type = hostrpc::client<N, hostrpc::x64_x64_bitmap_types,
                                      decltype(cp), hostrpc::fill_nop,
                                      hostrpc::use_nop, hostrpc::nop_stepper>;

  client_type c = {cp, inbox, outbox, active, remote_buffer, local_buffer};

  for (;;)
    {
      c.rpc_invoke<true>(nullptr);
      c.rpc_invoke<false>(nullptr);
    }
}

extern "C" void instantiate_try_garbage_collect_word_client(
    const hostrpc::mailbox_t<N> inbox, hostrpc::mailbox_t<N> outbox,
    hostrpc::slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active, uint64_t w)
{
  auto c = [](uint64_t i, uint64_t) -> uint64_t { return i; };
  hostrpc::try_garbage_collect_word<N, decltype(c)>(c, inbox, outbox, active,
                                                    w);
}
