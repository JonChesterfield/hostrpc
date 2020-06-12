#include "client_impl.hpp"
#define N 128

struct copy_functor_nop
    : public hostrpc::copy_functor_interface<copy_functor_nop>
{
};

using client_type =
    hostrpc::client_impl<N, copy_functor_nop, hostrpc::fill_nop,
                         hostrpc::use_nop, hostrpc::nop_stepper>;

extern "C" __attribute__((noinline)) void client_instance_direct(client_type& c)
{
  for (;;)
    {
      c.rpc_invoke<true>(nullptr);
      c.rpc_invoke<false>(nullptr);
    }
}

extern "C" __attribute__((noinline)) void client_instance_from_components(
    hostrpc::slot_bitmap_all_svm<N> inbox,
    hostrpc::slot_bitmap_all_svm<N> outbox,
    hostrpc::slot_bitmap_device<N> active, hostrpc::page_t* remote_buffer,
    hostrpc::page_t* local_buffer)
{
  client_type c = {inbox, outbox, active, remote_buffer, local_buffer};
  client_instance_direct(c);
}

void sink(client_type*);

extern "C" __attribute__((noinline)) void client_instance_from_words(
    void** from)
{
  client_type c;
  c.deserialize(from);
  client_instance_direct(c);
}

extern "C" __attribute__((noinline)) void client_instance_from_cast(void* from)
{
  client_type* c = reinterpret_cast<client_type*>(from);
  client_instance_direct(*c);
}
