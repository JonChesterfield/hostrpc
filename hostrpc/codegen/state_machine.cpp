#include "../detail/state_machine.hpp"

using state_machine_t =
    hostrpc::state_machine_impl<hostrpc::page_t, uint32_t,
                                hostrpc::size_compiletime<128>,
                                hostrpc::counters::client_nop, false>;

extern "C"
void partial_port()
{
  using namespace hostrpc;
  auto threads = platform::active_threads();
  size_compiletime<128> sz;
  state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

  state_machine_t::partial_port_t<1> p1 = s.rpc_open_partial_port(threads);

  state_machine_t::partial_port_t<0> p0 = s.rpc_port_apply(threads, cxx::move(p1), [](port_t, page_t *) {});

  s.rpc_close_port(threads, cxx::move(p0));

}

extern "C"
void typed_port_via_wait()
{
  using namespace hostrpc;
  auto threads = platform::active_threads();
  size_compiletime<128> sz;
  state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

  auto p00 = s.rpc_open_typed_port<0, 0>(threads);

  auto p01 = s.rpc_port_apply(threads, cxx::move(p00), [](port_t, page_t *) {});

  state_machine_t::typed_port_t<1, 1> p11 = s.rpc_port_wait(threads, cxx::move(p01));

  auto p10 = s.rpc_port_apply(threads, cxx::move(p11), [](port_t, page_t *) {});
  
  s.rpc_close_port(threads, cxx::move(p10));
}

extern "C"
void typed_port_via_query()
{
  using namespace hostrpc;
  auto threads = platform::active_threads();
  size_compiletime<128> sz;
  state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

  auto p00 = s.rpc_open_typed_port<0, 0>(threads);

  auto p01 = s.rpc_port_apply(threads, cxx::move(p00), [](port_t, page_t *) {});

  state_machine_t::typed_port_t<1, 1> p11;

  for (;;)
    {
      auto an_either = s.rpc_port_query(threads, cxx::move(p01));
      if (an_either)
        {
          auto a_maybe = an_either.on_true();
          if (a_maybe)
            {
              auto a = a_maybe.value();
              p01 = cxx::move(a);
            }
          else
            {
              __builtin_unreachable();
            }
        }
      else
        {
          auto a_maybe = an_either.on_false();
          if (a_maybe)
            {
              auto a = a_maybe.value();
              p11 = cxx::move(a);
              break;
            }
          else
            {
              __builtin_unreachable();
            }

        }
    }

  auto p10 = s.rpc_port_apply(threads, cxx::move(p11), [](port_t, page_t *) {});
  
  s.rpc_close_port(threads, cxx::move(p10));
}
