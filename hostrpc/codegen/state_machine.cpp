#include "../detail/state_machine.hpp"

struct buffer_ty
{
  int x;
};

using state_machine_t =
    hostrpc::state_machine_impl<buffer_ty, uint32_t,
                                hostrpc::size_compiletime<128>, false>;

using namespace hostrpc;

auto master_lane()
{
  auto threads = platform::active_threads();
  return platform::get_master_lane_id(threads);
}

auto is_master_lane()
{
  auto threads = platform::active_threads();
  return platform::is_master_lane(threads);
}

bool is_master_when_all_active()
{
  // works through implicit conversion to the underlying runtime value
  auto all = platform::all_threads_active_constant();
  return platform::is_master_lane(all);
}

extern "C"
{
  void open_and_close_partial_port(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p = s.rpc_open_partial_port(threads);
    s.rpc_close_port(threads, cxx::move(p));
  }

  void open_and_close_typed_port_lo(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p = s.rpc_open_typed_port<0, 0>(threads);
    s.rpc_close_port(threads, cxx::move(p));
  }

  void open_and_close_typed_port_hi(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p = s.rpc_open_typed_port<1, 1>(threads);
    s.rpc_close_port(threads, cxx::move(p));
  }

  void try_open_and_close_partial_port(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto m = s.rpc_try_open_partial_port(threads);
    if (m)
      {
        s.rpc_close_port(threads, m.value());
      }
  }

  void try_open_and_close_typed_port_lo(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto m = s.rpc_try_open_typed_port<0, 0>(threads);
    if (m)
      {
        s.rpc_close_port(threads, m.value());
      }
  }

  void try_open_and_close_typed_port_hi(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto m = s.rpc_try_open_typed_port<1, 1>(threads);
    if (m)
      {
        s.rpc_close_port(threads, m.value());
      }
  }

  void open_and_close_typed_port_lo_via_partial(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p0 = s.rpc_open_typed_port<0, 0>(threads);
    state_machine_t::partial_port_t<1> p1 = p0;
    s.rpc_close_port(threads, cxx::move(p1));
  }

  void open_and_close_typed_port_hi_via_partial(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p0 = s.rpc_open_typed_port<1, 1>(threads);
    state_machine_t::partial_port_t<1> p1 = p0;
    s.rpc_close_port(threads, cxx::move(p1));
  }

  void open_and_close_typed_port_lo_via_partial_after_apply(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p00 = s.rpc_open_typed_port<0, 0>(threads);
    auto p01 =
        s.rpc_port_apply(threads, cxx::move(p00), [](uint32_t, buffer_ty *) {});
    state_machine_t::partial_port_t<0> pU = p01;
    s.rpc_close_port(threads, cxx::move(pU));
  }

  void open_and_close_typed_port_hi_via_partial_after_apply(state_machine_t &s)
  {
    auto threads = platform::active_threads();
    auto p11 = s.rpc_open_typed_port<1, 1>(threads);
    auto p10 =
        s.rpc_port_apply(threads, cxx::move(p11), [](uint32_t, buffer_ty *) {});
    state_machine_t::partial_port_t<0> pU = p10;
    s.rpc_close_port(threads, cxx::move(pU));
  }
}

template <unsigned S>
static state_machine_t::partial_port_t<S> partial_S_nop_via_typed_port(
    state_machine_t::partial_port_t<S> &&p)
{
  hostrpc::either<state_machine_t::typed_port_t<0, S ? 0 : 1>,
                  state_machine_t::typed_port_t<1, S ? 1 : 0>>
      either = p;

  if (either)
    {
      auto maybe = either.on_true();
      if (maybe)
        {
          state_machine_t::typed_port_t<0, S ? 0 : 1> p00 = maybe.value();
          return p00;
        }
      else
        {
          __builtin_unreachable();
        }
    }
  else
    {
      auto maybe = either.on_false();
      if (maybe)
        {
          state_machine_t::typed_port_t<1, S ? 1 : 0> p11 = maybe.value();
          return p11;
        }
      else
        {
          __builtin_unreachable();
        }
    }
}

auto partial_0_nop_via_typed_port(state_machine_t::partial_port_t<0> &&p)
{
  return partial_S_nop_via_typed_port<0>(cxx::move(p));
}

auto partial_1_nop_via_typed_port(state_machine_t::partial_port_t<1> &&p)
{
  return partial_S_nop_via_typed_port<1>(cxx::move(p));
}

template <unsigned IA, unsigned OA, unsigned IB, unsigned OB, typename T>
hostrpc::either<state_machine_t::typed_port_t<!IA, OA>,
                state_machine_t::typed_port_t<!IB, OB>>
wait_inbox_on_either(
    T active_threads, state_machine_t &s,
    hostrpc::either<state_machine_t::typed_port_t<IA, OA>,
                    state_machine_t::typed_port_t<IB, OB>> &&port)
{
  // Looks functional, seeking a way to collapse the two inbox read loops
  // Tricky in that they're each waiting for a specific value when really
  if (!port)
    {
      return wait_inbox_on_either(active_threads, s, port.invert()).invert();
    }
  else
    {
      if (typename state_machine_t::typed_port_t<IA, OA>::maybe maybe =
              port.on_true())
        {
          state_machine_t::typed_port_t<IA, OA> first = maybe.value();
          state_machine_t::typed_port_t<!IA, OA> waited =
              s.rpc_port_wait(active_threads, cxx::move(first));

          return hostrpc::either<
              state_machine_t::typed_port_t<!IA, OA>,
              state_machine_t::typed_port_t<!IB, OB>>::Left(cxx::move(waited));
        }
      else
        {
          __builtin_unreachable();
        }
    }
}

auto wait_inbox_on_either(
    state_machine_t &s,
    hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                    state_machine_t::typed_port_t<1, 0>> &&port)
{
  auto threads = platform::active_threads();
  return wait_inbox_on_either(threads, s, cxx::move(port));
}

auto apply_partial_port(state_machine_t &s,
                        state_machine_t::partial_port_t<1> &&p0,
                        void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  return s.rpc_port_apply(threads, cxx::move(p0),
                          [=](uint32_t, buffer_ty *b) { func(b); });
}

auto apply_typed_port_lo(state_machine_t &s,
                         state_machine_t::typed_port_t<0, 0> &&p0,
                         void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  return s.rpc_port_apply(threads, cxx::move(p0),
                          [=](uint32_t, buffer_ty *b) { func(b); });
}

auto apply_typed_port_hi(state_machine_t &s,
                         state_machine_t::typed_port_t<1, 1> &&p0,
                         void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  return s.rpc_port_apply(threads, cxx::move(p0),
                          [=](uint32_t, buffer_ty *b) { func(b); });
}

auto on_element_partial_port(state_machine_t &s,
                             state_machine_t::partial_port_t<1> &&p0,
                             void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  s.rpc_port_on_element(threads, p0, [=](uint32_t, buffer_ty *b) { func(b); });
  return cxx::move(p0);
}

auto on_element_typed_port_lo(state_machine_t &s,
                              state_machine_t::typed_port_t<0, 0> &&p0,
                              void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  s.rpc_port_on_element(threads, p0, [=](uint32_t, buffer_ty *b) { func(b); });
  return cxx::move(p0);
}

auto on_element_typed_port_hi(state_machine_t &s,
                              state_machine_t::typed_port_t<1, 1> &&p0,
                              void func(buffer_ty *))
{
  auto threads = platform::active_threads();
  s.rpc_port_on_element(threads, p0, [=](uint32_t, buffer_ty *b) { func(b); });
  return cxx::move(p0);
}

auto wait_partial_port(state_machine_t &s,
                       state_machine_t::partial_port_t<0> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_wait(threads, cxx::move(p0));
}

auto wait_typed_port_lo(state_machine_t &s,
                        state_machine_t::typed_port_t<1, 0> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_wait(threads, cxx::move(p0));
}

auto wait_typed_port_hi(state_machine_t &s,
                        state_machine_t::typed_port_t<0, 1> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_wait(threads, cxx::move(p0));
}

auto query_partial_port(state_machine_t &s,
                        state_machine_t::partial_port_t<0> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_query(threads, cxx::move(p0));
}

auto query_typed_port_lo(state_machine_t &s,
                         state_machine_t::typed_port_t<1, 0> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_query(threads, cxx::move(p0));
}

auto query_typed_port_hi(state_machine_t &s,
                         state_machine_t::typed_port_t<0, 1> &&p0)
{
  auto threads = platform::active_threads();
  return s.rpc_port_query(threads, cxx::move(p0));
}

auto visit_either(state_machine_t &s, state_machine_t::typed_port_t<0, 1> &&p0)
{
  auto threads = platform::active_threads();
  hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                  state_machine_t::typed_port_t<1, 1>>
      either = s.rpc_port_query(threads, cxx::move(p0));

  auto nop = either.visit2(
      [](state_machine_t::typed_port_t<0, 1> &&port) {
        return cxx::move(port);
      },
      [](state_machine_t::typed_port_t<1, 1> &&port) {
        return cxx::move(port);
      });

  hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                  state_machine_t::typed_port_t<1, 0>>
      conditional_apply = nop.visit2(
          [](state_machine_t::typed_port_t<0, 1> &&port) {
            return cxx::move(port);
          },
          [&](state_machine_t::typed_port_t<1, 1> &&port) {
            return s.rpc_port_apply(threads, cxx::move(port),
                                    [](uint32_t, buffer_ty *) {});
          });

  hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                  state_machine_t::typed_port_t<0, 0>>
      conditional_wait = conditional_apply.visit2(
          [](state_machine_t::typed_port_t<0, 1> &&port) {
            return cxx::move(port);
          },
          [&](state_machine_t::typed_port_t<1, 0> &&port) {
            return s.rpc_port_wait(threads, cxx::move(port));
          });

  hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                  state_machine_t::typed_port_t<0, 1>>
      conditional_apply_again = conditional_wait.visit2(
          [](state_machine_t::typed_port_t<0, 1> &&port) {
            return cxx::move(port);
          },
          [&](state_machine_t::typed_port_t<0, 0> &&port) {
            return s.rpc_port_apply(threads, cxx::move(port),
                                    [](uint32_t, buffer_ty *) {});
          });

  state_machine_t::typed_port_t<0, 1> flatten =
      conditional_apply_again.on_true_and_false();

  s.rpc_close_port(threads, cxx::move(flatten));
}

auto maybe_either(state_machine_t &s, state_machine_t::typed_port_t<0, 1> &&p0)
{
  auto threads = platform::active_threads();
  hostrpc::either<state_machine_t::typed_port_t<0, 1>,
                  state_machine_t::typed_port_t<1, 1>>
      either = s.rpc_port_query(threads, cxx::move(p0));

  return hostrpc::either<
      state_machine_t::typed_port_t<0, 1>,
      state_machine_t::typed_port_t<1, 1>>::maybe(cxx::move(either));
}

state_machine_t::partial_port_t<1> wait_via_query_partial_port(
    state_machine_t &s, state_machine_t::partial_port_t<0> &&p0)
{
  auto threads = platform::active_threads();

  for (;;)
    {
      auto either = s.rpc_port_query(threads, cxx::move(p0));
      if (either)
        {
          auto maybe = either.on_true();
          if (maybe)
            {
              state_machine_t::partial_port_t<0> &&p1 = maybe.value();
              p0 = cxx::move(p1);
              continue;
            }
          else
            {
              __builtin_unreachable();
            }
        }
      else
        {
          auto maybe = either.on_false();
          if (maybe)
            {
              state_machine_t::partial_port_t<1> &&p1 = maybe.value();
              return cxx::move(p1);
            }
          else
            {
              __builtin_unreachable();
            }
        }
    }
}

state_machine_t::typed_port_t<0, 0> wait_via_query_typed_port_lo(
    state_machine_t &s, state_machine_t::typed_port_t<1, 0> &&p0)
{
  auto threads = platform::active_threads();

  for (;;)
    {
      auto either = s.rpc_port_query(threads, cxx::move(p0));
      if (either)
        {
          auto maybe = either.on_true();
          if (maybe)
            {
              state_machine_t::typed_port_t<1, 0> &&p1 = maybe.value();
              p0 = cxx::move(p1);
              continue;
            }
          else
            {
              __builtin_unreachable();
            }
        }
      else
        {
          auto maybe = either.on_false();
          if (maybe)
            {
              state_machine_t::typed_port_t<0, 0> &&p1 = maybe.value();
              return cxx::move(p1);
            }
          else
            {
              __builtin_unreachable();
            }
        }
    }
}

state_machine_t::typed_port_t<1, 1> wait_via_query_typed_port_hi(
    state_machine_t &s, state_machine_t::typed_port_t<0, 1> &&p0)
{
  auto threads = platform::active_threads();

  for (;;)
    {
      auto either = s.rpc_port_query(threads, cxx::move(p0));
      if (either)
        {
          auto maybe = either.on_true();
          if (maybe)
            {
              state_machine_t::typed_port_t<0, 1> &&p1 = maybe.value();
              p0 = cxx::move(p1);
              continue;
            }
          else
            {
              __builtin_unreachable();
            }
        }
      else
        {
          auto maybe = either.on_false();
          if (maybe)
            {
              state_machine_t::typed_port_t<1, 1> &&p1 = maybe.value();
              return cxx::move(p1);
            }
          else
            {
              __builtin_unreachable();
            }
        }
    }
}

extern "C"
{
  void partial_port()
  {
    auto threads = platform::active_threads();
    size_compiletime<128> sz;
    state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

    state_machine_t::partial_port_t<1> p1 = s.rpc_open_partial_port(threads);

    state_machine_t::partial_port_t<0> p0 =
        s.rpc_port_apply(threads, cxx::move(p1), [](uint32_t, buffer_ty *) {});

    s.rpc_close_port(threads, cxx::move(p0));
  }

  void typed_port_via_wait()
  {
    using namespace hostrpc;
    auto threads = platform::active_threads();
    size_compiletime<128> sz;
    state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

    auto p00 = s.rpc_open_typed_port<0, 0>(threads);

    auto p01 =
        s.rpc_port_apply(threads, cxx::move(p00), [](uint32_t, buffer_ty *) {});

    state_machine_t::typed_port_t<1, 1> p11 =
        s.rpc_port_wait(threads, cxx::move(p01));

    auto p10 =
        s.rpc_port_apply(threads, cxx::move(p11), [](uint32_t, buffer_ty *) {});

    s.rpc_close_port(threads, cxx::move(p10));
  }

  void typed_port_via_query()
  {
    using namespace hostrpc;
    auto threads = platform::active_threads();
    size_compiletime<128> sz;
    state_machine_t s(sz, {nullptr}, {nullptr}, {nullptr}, nullptr);

    auto p00 = s.rpc_open_typed_port<0, 0>(threads);

    auto p01 =
        s.rpc_port_apply(threads, cxx::move(p00), [](uint32_t, buffer_ty *) {});

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

    auto p10 =
        s.rpc_port_apply(threads, cxx::move(p11), [](uint32_t, buffer_ty *) {});

    s.rpc_close_port(threads, cxx::move(p10));
  }
}
