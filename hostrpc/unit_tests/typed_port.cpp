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
  static constexpr partial_port_t<S> make(uint32_t v, bool s)
  {
    return {v, s};
  }

  template <unsigned I, unsigned O>
  static constexpr void drop(typed_port_t<I, O>&& port)
  {
    port.drop();
  }

  template <unsigned S>
  static constexpr void drop(partial_port_t<S>&& port)
  {
    port.drop();
  }

  template <unsigned I, unsigned O>
  static auto typed_to_partial(typed_port_t<I, O>&& port)
  {
    using info = hostrpc::traits::typed_to_partial_trait<test_state_machine,
                                                         typed_port_t<I, O>>;
    return typename info::type((uint32_t)port, info::state);
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

MAIN_MODULE()
{
  DEPENDS(create_and_immediately_destroy);
  DEPENDS(maybe);
  DEPENDS(conversions);
}
