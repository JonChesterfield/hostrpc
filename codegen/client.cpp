#include "../client_impl.hpp"

using SZ = hostrpc::size_compiletime<128>;

struct copy_functor_nop
    : public hostrpc::copy_functor_interface<copy_functor_nop>
{
};

using client_type =
    hostrpc::client_impl<SZ, copy_functor_nop, hostrpc::fill_nop,
                         hostrpc::use_nop, hostrpc::nop_stepper>;

extern "C" __attribute__((noinline)) void client_instance_direct(client_type& c)
{
  for (;;)
    {
      c.rpc_invoke<true>(nullptr, nullptr);
      c.rpc_invoke<false>(nullptr, nullptr);
    }
}

extern "C" __attribute__((noinline)) void client_instance_from_components(
    SZ sz, hostrpc::slot_bitmap_all_svm inbox,
    hostrpc::slot_bitmap_all_svm outbox, hostrpc::slot_bitmap_device active,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  client_type c = {sz, inbox, outbox, active, remote_buffer, local_buffer};
  client_instance_direct(c);
}

void sink(client_type*);

extern "C" __attribute__((noinline)) void client_instance_from_cast(void* from)
{
  client_type* c = reinterpret_cast<client_type*>(from);
  client_instance_direct(*c);
}

extern "C" __attribute__((noinline)) void client_instance_from_aliasing(
    void* from)
{
  using aliasing_client_type = __attribute__((__may_alias__)) client_type;
  aliasing_client_type* c = reinterpret_cast<aliasing_client_type*>(from);
  client_instance_direct(*c);
}
