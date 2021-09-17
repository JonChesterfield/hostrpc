#include "../detail/client_impl.hpp"
#include "../detail/platform_detect.hpp"

#include "../allocator.hpp"

using SZ = hostrpc::size_compiletime<128>;

using client_type =
    hostrpc::client<uint64_t, SZ, hostrpc::counters::client_nop>;

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_invoke_direct(client_type& c)
{
  for (;;)
    {
      hostrpc::fill_nop fill;
      hostrpc::use_nop use;
      c.rpc_invoke(fill, use);
      c.rpc_invoke(fill);
    }
}

extern "C" __attribute__((always_inline)) HOSTRPC_ANNOTATE void
client_instance_invoke_via_port(client_type& c)
{
  auto active_threads = platform::active_threads();
  uint32_t p = c.rpc_open_port(active_threads);
  c.rpc_port_send(active_threads, p, hostrpc::fill_nop{});
  c.rpc_port_wait_for_result(active_threads, p);
  c.rpc_port_recv(active_threads, p, hostrpc::use_nop{});
  c.rpc_close_port(active_threads, p);
}

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_from_components(SZ sz, client_type::inbox_t inbox,
                                client_type::outbox_t outbox,
                                client_type::lock_t active,
                                client_type::staging_t staging,
                                hostrpc::page_t* shared_buffer)
{
  client_type c = {sz, active, inbox, outbox, staging, shared_buffer};
  client_instance_invoke_direct(c);
}

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_via_port_from_components(SZ sz, client_type::inbox_t inbox,
                                         client_type::outbox_t outbox,
                                         client_type::lock_t active,
                                         client_type::staging_t staging,
                                         hostrpc::page_t* shared_buffer)
{
  client_type c = {sz, active, inbox, outbox, staging, shared_buffer};
  client_instance_invoke_via_port(c);
}

HOSTRPC_ANNOTATE void sink(client_type*);

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_from_cast(void* from)
{
  client_type* c = reinterpret_cast<client_type*>(from);
  client_instance_invoke_direct(*c);
}

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_from_aliasing(void* from)
{
  using aliasing_client_type = __attribute__((__may_alias__)) client_type;
  aliasing_client_type* c = reinterpret_cast<aliasing_client_type*>(from);
  client_instance_invoke_direct(*c);
}
