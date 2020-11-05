#include "../detail/client_impl.hpp"

#include "../allocator.hpp"

using SZ = hostrpc::size_compiletime<128>;

struct copy_functor_nop
    : public hostrpc::copy_functor_interface<copy_functor_nop>
{
};

using client_type =
    hostrpc::client_impl<uint32_t, SZ, copy_functor_nop, hostrpc::fill_nop,
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
    SZ sz, client_type::inbox_t inbox, client_type::outbox_t outbox,
    client_type::lock_t active, client_type::staging_t staging,
    hostrpc::page_t* remote_buffer, hostrpc::page_t* local_buffer)
{
  client_type c = {sz,      active,        inbox,       outbox,
                   staging, remote_buffer, local_buffer};
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
