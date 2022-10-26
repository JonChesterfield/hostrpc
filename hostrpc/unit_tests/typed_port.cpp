#include "../detail/state_machine.hpp"
#include "EvilUnit.h"
#include "../detail/maybe.hpp"

namespace
{
// define a typed_port_t that can be constructed outside of state machine
struct typed_port_friend;

template <unsigned I, unsigned O>
using typed_port_t = hostrpc::typed_port_impl_t<typed_port_friend, I, O>;

struct typed_port_friend
{
  template <unsigned I, unsigned O>
  static constexpr typed_port_t<I, O> make(uint32_t v)
  {
    return {v};
  }

  template <unsigned I, unsigned O>
  static constexpr void drop(typed_port_t<I, O>&& port)
  {
    port.drop();
  }
};

template <unsigned I, unsigned O>
typed_port_t<I, O> constexpr make(uint32_t v)
{
  return typed_port_friend::make<I, O>(v);
}

template <unsigned I, unsigned O>
constexpr void drop(typed_port_t<I, O>&& port)
{
  return typed_port_friend::drop(hostrpc::cxx::move(port));
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
  }

  TEST("default constructed warns on conversion to uint32")
  {
    auto tmp = typed_port_t<0, 0>{};
    // uint32_t v = tmp; (void)v;
  }

  TEST("non-default constructed can be converted to uint32 without consuming")
  {
    auto tmp = make<0, 0>(10);
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


static MODULE(maybe)
{
  using namespace hostrpc;
  TEST("hack default")
    {
      typed_port_t<0,1>::maybe val {{}, false};
      if (val)
        {
          typed_port_t<0, 1> tmp = val;
          drop(cxx::move(tmp));
        }      
    }

  TEST("hack non-default")
    {
      typed_port_t<0,1>::maybe val {42, false};
      if (val)
        {
          typed_port_t<0, 1> tmp = val;
          CHECK(tmp == 42);
          drop(cxx::move(tmp));
        }      
    }

  
  
}

MAIN_MODULE() { DEPENDS(create_and_immediately_destroy); DEPENDS(maybe);}
