#include "../detail/maybe.hpp"
#include "../detail/state_machine.hpp"
#include "EvilUnit.h"

namespace
{
// define a typed_port_t that can be constructed outside of state machine
struct test_state_machine;

template <unsigned I, unsigned O>
using typed_port_t = hostrpc::typed_port_impl_t<test_state_machine, I, O>;

template <unsigned S>
using partial_port_t = hostrpc::partial_port_impl_t<test_state_machine, S>;

struct test_state_machine
{
  static_assert(hostrpc::traits::traits_consistent<test_state_machine>());

  template <unsigned I, unsigned O>
  static constexpr typed_port_t<I, O> make(uint32_t v)
  {
    return {v};
  }

  template <unsigned S>
  static partial_port_t<S> make(uint32_t v, bool s)
  {
    return {v, s};
  }

  template <unsigned S>
  static partial_port_t<S> make(hostrpc::cxx::tuple<uint32_t, bool> tup)
  {
    return {tup};
  }

  template <unsigned I, unsigned O>
  static constexpr void drop(typed_port_t<I, O>&& port)
  {
    port.kill();
  }

  template <unsigned S>
  static constexpr void drop(partial_port_t<S>&& port)
  {
    port.kill();
  }

  template <unsigned I, unsigned O>
  static
      typename hostrpc::traits::typed_to_partial_trait<test_state_machine,
                                                       typed_port_t<I, O>>::type
      typed_to_partial(typed_port_t<I, O>&& port HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = port;
    port.kill();
    return {v,
            hostrpc::traits::typed_to_partial_trait<test_state_machine,
                                                    typed_port_t<I, O>>::state};
  }

  template <bool OutboxState, unsigned S>
  static HOSTRPC_RETURN_UNKNOWN hostrpc::maybe<
      uint32_t, typename hostrpc::traits::partial_to_typed_trait<
                    test_state_machine, partial_port_t<S>, OutboxState>::type>
  partial_to_typed(partial_port_t<S>&& port HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = port.value;
    bool state = port.state;

    if (OutboxState == state)
      {
        port.kill();
        return {v};
      }
    else
      {
        drop(hostrpc::cxx::move(port));
        return {};
      }
  }
};

template <unsigned I, unsigned O>
typed_port_t<I, O> constexpr make(uint32_t v)
{
  return test_state_machine::make<I, O>(v);
}

template <unsigned I, unsigned O>
constexpr void drop(typed_port_t<I, O>&& port)
{
  return test_state_machine::drop(hostrpc::cxx::move(port));
}

template <unsigned S>
partial_port_t<S> constexpr make(uint32_t v, bool s)
{
  return test_state_machine::make<S>(v, s);
}

template <unsigned S>
partial_port_t<S> make(hostrpc::cxx::tuple<uint32_t, bool> tup)
{
  return test_state_machine::make<S>(tup);
}

template <unsigned S>
constexpr void drop(partial_port_t<S>&& port)
{
  return test_state_machine::drop(hostrpc::cxx::move(port));
}

}  // namespace

MODULE(create_and_immediately_destroy)
{
  using namespace hostrpc;

  TEST("default constructed can be dropped without warning")
  {
    typed_port_t<0, 0>{};
    typed_port_t<0, 1>{};
    typed_port_t<1, 0>{};
    typed_port_t<1, 1>{};

    partial_port_t<0>{};
    partial_port_t<1>{};
  }

  TEST("default constructed warns on conversion to uint32")
  {
    auto tmp = typed_port_t<0, 0>{};
    // uint32_t v = tmp; (void)v;
  }

  TEST("default constructed warns on conversion to uint32")
  {
    auto tmp = partial_port_t<0>{};
    // uint32_t v = tmp; (void)v;
  }

  TEST("non-default constructed can be converted to uint32 without consuming")
  {
    auto tmp = make<0, 0>(10);
    CHECK(10 == tmp);
    CHECK(20 != tmp);
    drop(cxx::move(tmp));
  }

  TEST("non-default constructed can be converted to uint32 without consuming")
  {
    auto tmp = make<0>(10, true);
    CHECK(10 == tmp);
    CHECK(20 != tmp);
    drop(cxx::move(tmp));
  }

  TEST("make partial from a tuple")
  {
    cxx::tuple<uint32_t, bool> tup = {12, false};
    partial_port_t<1> tmp = make<1>(tup);
    CHECK(tmp == 12);
    drop(cxx::move(tmp));
  }

  TEST("move initialization ok")
  {
    typed_port_t<1, 1> var0 = make<1, 1>(12);
    CHECK(12 == var0);

    typed_port_t<1, 1> var1 = cxx::move(var0);
    // CHECK(12 == var0);
    CHECK(12 == var1);
    drop(cxx::move(var1));
  }

  TEST("move initialization ok")
  {
    partial_port_t<1> var0 = make<1>(12, false);
    CHECK(12 == var0);

    partial_port_t<1> var1 = cxx::move(var0);
    // CHECK(12 == var0);
    CHECK(12 == var1);
    drop(cxx::move(var1));
  }

  TEST("move assignment by cast ok")
  {
    typed_port_t<1, 1> var0;
    typed_port_t<1, 1> var1 = make<1, 1>(10);

    // (void) (var0==4);
    (void)(var1 == 4);

    var0 = static_cast<typed_port_t<1, 1>&&>(var1);

    (void)(var0 == 4);

    drop(cxx::move(var0));
  }

  TEST("move assignment by cxx::move ok")
  {
    // had to do bad things with std::move to make that work
    typed_port_t<1, 1> var0;
    typed_port_t<1, 1> var1 = make<1, 1>(10);

    // (void) (var0==4);
    CHECK(var1 == 10);

    var0 = cxx::move(var1);
    CHECK(var0 == 10);
    // CHECK(var1 == 10);

    drop(cxx::move(var0));
    // drop(cxx::move(var1));
  }

  TEST("move twice")
  {
    typed_port_t<0, 1> var0 = cxx::move(make<0, 1>(20));
    (void)(var0 == 20);
    typed_port_t<0, 1> var1 = cxx::move(cxx::move(var0));
    // (void)(var0 == 20);
    drop(cxx::move(var1));
  }

  TEST("00") { drop(make<0, 0>(43)); }
  TEST("01") { drop(make<0, 1>(43)); }
  TEST("10") { drop(make<1, 0>(44)); }
  TEST("11") { drop(make<1, 1>(45)); }
}

static MODULE(conversions)
{
  TEST("typed to partial")
  {
    auto tmp = make<0, 0>(10);
    auto partial =
        test_state_machine::typed_to_partial(hostrpc::cxx::move(tmp));
    CHECK(partial == 10);
    drop(hostrpc::cxx::move(partial));
  }

  TEST("typed to partial")
  {
    auto tmp = make<1, 1>(10);
    auto partial =
        test_state_machine::typed_to_partial(hostrpc::cxx::move(tmp));
    CHECK(partial == 10);
    drop(hostrpc::cxx::move(partial));
  }

  TEST("partial to typed, S0, directly drop")
  {
    for (int i = 0; i < 2; i++)
      {
        bool state = i;

        partial_port_t<0> start = make<0>(3, state);
        typed_port_t<0, 1>::maybe typed =
            test_state_machine::partial_to_typed<true>(
                hostrpc::cxx::move(start));

        if (typed)
          {
            auto v = typed.value();
            drop(hostrpc::cxx::move(v));
          }
      }
  }

  TEST("partial to typed, S 0")
  {
    for (int i = 0; i < 2; i++)
      {
        bool state = i;

        partial_port_t<0> start = make<0>(20, state);
        partial_port_t<0> regen;

        if (start.outbox<true>())
          {
            typed_port_t<0, 1>::maybe typed =
                test_state_machine::partial_to_typed<true>(
                    hostrpc::cxx::move(start));
            if (typed)
              {
                typed_port_t<0, 1> port = typed.value();
                regen = test_state_machine::typed_to_partial(
                    hostrpc::cxx::move(port));
              }
            else
              {
                __builtin_unreachable();
              }
          }
        else
          {
            typed_port_t<1, 0>::maybe typed =
                test_state_machine::partial_to_typed<false>(
                    hostrpc::cxx::move(start));
            if (typed)
              {
                typed_port_t<1, 0> port = typed.value();
                regen = test_state_machine::typed_to_partial(
                    hostrpc::cxx::move(port));
              }
            else
              {
                __builtin_unreachable();
              }
          }
        CHECK(regen == 20);
        drop(hostrpc::cxx::move(regen));
      }
  }

  TEST("partial to typed, S 1")
  {
    for (int i = 0; i < 2; i++)
      {
        bool state = i;

        partial_port_t<1> start = make<1>(30, state);
        partial_port_t<1> regen;

        if (start.outbox<true>())
          {
            typed_port_t<1, 1>::maybe typed =
                test_state_machine::partial_to_typed<true>(
                    hostrpc::cxx::move(start));
            if (typed)
              {
                typed_port_t<1, 1> port = typed.value();
                regen = test_state_machine::typed_to_partial(
                    hostrpc::cxx::move(port));
              }
            else
              {
                __builtin_unreachable();
              }
          }
        else
          {
            typed_port_t<0, 0>::maybe typed =
                test_state_machine::partial_to_typed<false>(
                    hostrpc::cxx::move(start));
            if (typed)
              {
                typed_port_t<0, 0> port = typed.value();
                regen = test_state_machine::typed_to_partial(
                    hostrpc::cxx::move(port));
              }
            else
              {
                __builtin_unreachable();
              }
          }
        CHECK(regen == 30);
        drop(hostrpc::cxx::move(regen));
      }
  }
}

static MODULE(maybe)
{
  using namespace hostrpc;
  TEST("maybe default")
  {
    typed_port_t<0, 1>::maybe val;
    val.unknown();
    if (val)
      {
        val.unconsumed();
        typed_port_t<0, 1> tmp = val;
        val.consumed();
        tmp.unconsumed();
        drop(cxx::move(tmp));
        tmp.consumed();
      }
    val.consumed();
  }

  TEST("maybe non-default")
  {
    typed_port_t<0, 1>::maybe val{42};
    val.unknown();
    if (val)
      {
        val.unconsumed();
        typed_port_t<0, 1> tmp = val;
        val.consumed();
        CHECK(tmp == 42);
        drop(cxx::move(tmp));
      }
    val.consumed();
  }
}

// annotating a non-const & with this typestate works, and gives you a function
// that can mutate the argument provided it leaves it in unconsumed state
template <typename F, unsigned I, unsigned O>
uint32_t raw_without_destroy(
    hostrpc::typed_port_impl_t<F, I, O> const& copy HOSTRPC_CONST_REF_ARG)
{
  uint32_t r = static_cast<uint32_t>(copy);
  return r;
}

static MODULE(const_reference)
{
  using namespace hostrpc;
  TEST("convert to raw without destroying it")
  {
    typed_port_t<0, 1> val = make<0, 1>(10);
    val.unconsumed();
    uint32_t should_fail = raw_without_destroy(val);
    CHECK(should_fail == 10);
    drop(cxx::move(val));
  }
}

MAIN_MODULE()
{
  DEPENDS(create_and_immediately_destroy);
  DEPENDS(conversions);
  DEPENDS(maybe);
  DEPENDS(const_reference);
}
