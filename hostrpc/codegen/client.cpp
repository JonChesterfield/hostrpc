#include "../detail/client_impl.hpp"
#include "../platform/detect.hpp"

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
client_instance_invoke_via_port_runtime(client_type& c)
{
  auto active_threads = platform::active_threads();
  auto p = c.rpc_open_port(active_threads);
  c.rpc_port_send(active_threads, p, hostrpc::fill_nop{});
  c.rpc_port_wait_for_result(active_threads, p);
  c.rpc_port_recv(active_threads, p, hostrpc::use_nop{});
  c.rpc_close_port(active_threads, p);
}

extern "C" __attribute__((always_inline)) HOSTRPC_ANNOTATE void
client_instance_invoke_via_port_all_active(client_type& c)
{
  auto active_threads = platform::all_threads_active_constant();
  auto p = c.rpc_open_port(active_threads);
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
  client_instance_invoke_via_port_runtime(c);
}

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

extern "C" HOSTRPC_ANNOTATE void reference(client_type& c)
{
  struct fill_line
  {
    // passing it as a reference gives bounds checking
    HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, uint32_t call_number,
                                     uint64_t (&element)[8])
    {
      (void)call_number;
      element[0] = element[1];
      element[6] = element[7];
    }
  };

  // opencl deduces the wrong type for fill line (__private qualifies it)
  c.rpc_invoke(hostrpc::make_apply<fill_line>(fill_line{}));
}
namespace hostrpc
{

// Would like to be able to pass either
// void operator()(port_t, uint32_t call_number, uint64_t *)
// or
// void operator()(port_t port, page_t *page)
// Disassembling via traits and counting number arguments is thwarted
// by opencl rejecting function pointers, even in unevaluated contexts



}  // namespace hostrpc

extern "C" HOSTRPC_ANNOTATE void pointer(client_type& c)
{
  struct fill_line
  {
    HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, uint32_t call_number,
                                     uint64_t (&element)[8])
    {
      (void)call_number;
      element[0] = element[1];
      element[6] = element[7];
    }
  };

  // opencl deduces the wrong type for fill line (__private qualifies it)
  auto a = hostrpc::make_apply<fill_line>(fill_line{});


  c.rpc_invoke(hostrpc::cxx::move(a));
}
