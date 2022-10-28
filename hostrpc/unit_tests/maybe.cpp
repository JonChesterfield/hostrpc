#include "EvilUnit.h"
#include "../detail/maybe.hpp"
#include <stdint.h>

using namespace hostrpc;

// shouldn't have to qualify this, but nevermind. Close enough
// a static factory function with the annotation doesn't help either
namespace {
  HOSTRPC_RETURN_UNKNOWN
maybe<float> from_func(float x)
{
  return maybe<float>(x);
}
}

MODULE(maybe)
{
  // tests are all looking for clean compilation, force some print output
  // as the test framework writes nothing if zero checks exist
  CHECK(true);
  
  using maybe_t = maybe<uint32_t>;

  TEST("default constructed")
    {
      maybe_t d;
      d.unknown();
      if (d)
        {
          CHECK(false); // not executed
          d.unconsumed();
          d.value();
          d.consumed();
        }
      d.consumed();
    }
  
  
  TEST("happy path, false")
  {
    maybe_t i;
    i.unknown();

    if (i)
      {
        i.unconsumed();
        uint32_t value = i;
        (void)value;
        i.consumed();
      }
    else
      {
        i.consumed();
      }

    i.consumed();
  }

  TEST("happy path, true")
  {
    maybe_t i(12);
    i.unknown();

    if (i)
      {
        i.unconsumed();
        uint32_t value = i;
        (void)value;
        i.consumed();
      }
    else
      {
        i.consumed();
      }

    i.consumed();
  }

  TEST("create and ignore is an error")
  {
    // maybe_t i;
    // maybe_t j {};
    // maybe_t v(24);
  }

  TEST("check and don't use")
  {
    maybe_t i;
    i.unknown();
    if (i)
      {
        i.unconsumed();
      }
    i.unknown();
    if (i)
      {
        uint32_t d = i;
        (void)d;
      }
    i.consumed();
  }

  TEST("converting to bool doesn't change state")
  {
    maybe_t i;
    bool v = static_cast<bool>(i);
    i.unknown();
    if (v)
      {
        i.unknown();
      }
    i.unknown();

    if (i)
      {
        uint32_t d = i;
        (void)d;
      }
    i.consumed();
  }

#if 0
  TEST("const maybe")
    {
      const maybe_t i(10);
      if (i)
        {
          uint32_t u = i;(void)u;
          i.consumed();
        }
      i.consumed();
    }
#endif

  TEST("normal maybe")
  {
    maybe_t i;
    if (i)
      {
        uint32_t u = i;
        (void)u;
        i.consumed();
      }
    i.consumed();
  }

  TEST("from function")
    {
      maybe<float> f = from_func(1.4);
      f.unknown();
      if (f) { float tmp = f; (void)tmp; }
    }
}

MAIN_MODULE()
{  
  DEPENDS(maybe);
}
