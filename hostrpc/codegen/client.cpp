#include "../detail/client_impl.hpp"
#include "../platform/detect.hpp"

#include "../allocator.hpp"

using SZ = hostrpc::size_compiletime<128>;

using client_type = hostrpc::client<hostrpc::page_t, uint64_t, SZ>;

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

extern "C" __attribute__((flatten)) HOSTRPC_ANNOTATE void client_open_any_port(
    client_type& c)
{
  using namespace hostrpc;
  auto active_threads = platform::active_threads();
  client_type::typed_port_t<0, 0>::maybe p0 =
      c.rpc_try_open_typed_port(active_threads);
  if (p0)
    {
      c.rpc_close_port(active_threads, p0.value());
    }
}

extern "C" __attribute__((flatten)) HOSTRPC_ANNOTATE void client_compiling_either(
    client_type& c)
{
  using namespace hostrpc;

  auto active_threads = platform::active_threads();
  client_type::typed_port_t<0, 0>::maybe p0 =
    c.rpc_try_open_typed_port(active_threads);
  
  if (p0)
    {
      client_type::typed_port_t<0, 0> p00(p0.value());
      client_type::typed_port_t<0, 1> p01 = c.rpc_port_send(active_threads,
                                                            cxx::move(p00),
                                                            [](uint32_t, hostrpc::page_t *) {});
      
      auto an_either = c.rpc_port_query<0,decltype(active_threads)>(active_threads, cxx::move(p01));
      if (an_either)
        {
          auto a_maybe = an_either.left([&](auto && port){c.rpc_close_port(active_threads, cxx::move(port));});
          if (a_maybe)
            {
              auto a = a_maybe.value();
              c.rpc_close_port(active_threads, cxx::move(a));
            }
        }
      else
        {
          auto a_maybe = an_either.right([&](auto && port){c.rpc_close_port(active_threads, cxx::move(port));});
          if (a_maybe)
            {
              auto a = a_maybe.value();
              c.rpc_close_port(active_threads, cxx::move(a));
            }
        }
    }
}

extern "C" __attribute__((always_inline)) HOSTRPC_ANNOTATE void
client_instance_invoke_via_typed_port_runtime(client_type& c)
{
  using namespace hostrpc;

  auto active_threads = platform::active_threads();
  client_type::typed_port_t<0, 0> p0 = c.rpc_open_typed_port(active_threads);
  client_type::typed_port_t<0, 1> p1 =
      c.rpc_port_send(active_threads, cxx::move(p0), hostrpc::fill_nop{});
  client_type::typed_port_t<1, 1> p2 =
      c.rpc_port_wait_for_result(active_threads, cxx::move(p1));
  client_type::typed_port_t<1, 0> p3 =
      c.rpc_port_wait(active_threads, cxx::move(p2), hostrpc::use_nop{});
  client_type::typed_port_t<0, 0> p4 =
      c.rpc_port_wait_until_available(active_threads, cxx::move(p3));
  c.rpc_close_port(active_threads, cxx::move(p4));
}

extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_from_components(SZ sz, client_type::inbox_t inbox,
                                client_type::outbox_t outbox,
                                client_type::lock_t active,
                                hostrpc::page_t* shared_buffer)
{
  client_type c = {sz, active, inbox, outbox, shared_buffer};
  client_instance_invoke_direct(c);
}

#if 0
extern "C" __attribute__((noinline)) HOSTRPC_ANNOTATE void
client_instance_via_port_from_components(SZ sz, client_type::inbox_t inbox,
                                         client_type::outbox_t outbox,
                                         client_type::lock_t active,
                                         hostrpc::page_t* shared_buffer)
{
  client_type c = {sz, active, inbox, outbox, shared_buffer};
  client_instance_invoke_via_port_runtime(c);
}
#endif

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

