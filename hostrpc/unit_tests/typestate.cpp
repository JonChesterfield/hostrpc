#include "EvilUnit.h"
#include <stdint.h>
#include "../detail/cxx.hpp"

#define HOSTRPC_RETURN_CONSUMED __attribute__((return_typestate(consumed)))
#define HOSTRPC_RETURN_UNKNOWN __attribute__((return_typestate(unknown)))

#define HOSTRPC_CONSUMED_ARG                                            \
  __attribute__((param_typestate(unconsumed)))  HOSTRPC_RETURN_CONSUMED

#define HOSTRPC_CREATED_RES __attribute__((return_typestate(unconsumed)))

#define HOSTRPC_CALL_ON_LIVE __attribute__((callable_when(unconsumed)))

#define HOSTRPC_CALL_ON_DEAD __attribute__((callable_when(consumed)))

#define HOSTRPC_CALL_ON_UNKNOWN __attribute__((callable_when(unknown)))

#define HOSTRPC_CONSUMABLE_CLASS __attribute__((consumable(unconsumed)))

#define HOSTRPC_SET_TYPESTATE(X) __attribute__((set_typestate(X)))


struct HOSTRPC_CONSUMABLE_CLASS  number
{

 HOSTRPC_RETURN_CONSUMED  number() : value(0) {}
 HOSTRPC_CREATED_RES  number(uint32_t v) : value(v) {}


 // shall not copy, sadly. Can't get it to reject copying consumed argument.
 number(const number &other) = delete; 
 number &operator=(const number & other) = delete;

  
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD
  number(number &&other HOSTRPC_CONSUMED_ARG)
      : value(other.value)
  {
    other.kill();
    def();
  }

  
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD number &operator=(
      number &&other HOSTRPC_CONSUMED_ARG)
  {
    value = other.value;
    other.kill();
    def();
    return *this;
  }

 
 // can't be default as that then ignores the annotation
 HOSTRPC_CALL_ON_DEAD  ~number() {}


 HOSTRPC_SET_TYPESTATE(consumed)
 HOSTRPC_CALL_ON_LIVE operator uint32_t()  const
  {
   //     this->kill();
    return value;
  }


 HOSTRPC_CALL_ON_LIVE HOSTRPC_SET_TYPESTATE(consumed) void drop() {}

 

  // not convinced test_typestate is useful for anything
  HOSTRPC_CALL_ON_DEAD void consumed() const {}
  HOSTRPC_CALL_ON_LIVE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN void unknown() const {}

  HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
 
 private:

  HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

  uint32_t value;
};



HOSTRPC_CREATED_RES
number function(number && x   __attribute__((param_typestate(unconsumed))))
                                    
{
  uint32_t val = static_cast<uint32_t>(x);
  x.kill();
  return number(val);
}

MODULE(create_and_immediately_destroy)
{
  TEST("default ctor, drop out of scope, ok")
    {
     number d;
   d.consumed();
    }

  TEST("non-default ctor, drop out of scope not ok")
    {
      number d(42);
      d.unconsumed();
      d.drop();
      d.consumed();
    }

  // couldn't get copy constructors to behave, sadly
#if 0
  TEST("default ctor, copy must fail on it")
    {
      number d;
      d.consumed();
      number e(d); // wanted an error here, can't get it
      e.unconsumed();
      e.drop();
    }

  TEST("copy ok")
    {
      number d(10);
      d.unconsumed();
      number e(d);
      d.consumed(); // hasn't been consumed by the copy ctor
      e.unconsumed(); // this is fine
      e.drop();
    }
#endif

  TEST("static cast drops it")
    {
      number d(10);
      d.unconsumed();
      uint32_t v = d;
      d.consumed();
      (void)v;
    }

  TEST("copy via uint32_t")
    {
      number d(16);
      d.unconsumed();
      number e(static_cast<uint32_t>(d));
      d.consumed();
      e.unconsumed();
      e.drop();
    }

  TEST("copy via conversion")
    {
      number d(16);
      d.unconsumed();
      number e((uint32_t)d); // does need to be explicit
      d.consumed();
      e.unconsumed();
      e.drop();
    }
  
  
  TEST("function can take rvalue")
    {
      number f = function( number{10} );
      f.unconsumed();
      f.drop();
    }

  TEST("copy")
    {
      number tmp = number(10);
      number res = function(number(static_cast<uint32_t>(tmp)));
      res.drop();

    }
  
  TEST("rvalue")
    {
      number tmp = number(10);
      number res = function(static_cast<number&&>(tmp));
      tmp.consumed();
      
      res.drop();
      
      
    }

}


template <typename T>
struct HOSTRPC_CONSUMABLE_CLASS maybe
{

 HOSTRPC_RETURN_UNKNOWN   
   maybe(T && payload, bool valid)
   :payload(hostrpc::cxx::forward<T>(payload)),
   valid(valid)
 {
 }

  HOSTRPC_CALL_ON_DEAD void consumed() const {}
  HOSTRPC_CALL_ON_LIVE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN void unknown() const {}


  HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
    HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}
    
   HOSTRPC_CALL_ON_UNKNOWN


     
  operator bool()  __attribute__ ((test_typestate(unconsumed)))
    {
      return valid;
    }
   #if 0
  operator bool()  const __attribute__ ((test_typestate(unconsumed)))
    {
      return valid;
    }
  #endif

  HOSTRPC_SET_TYPESTATE(consumed)
    HOSTRPC_CALL_ON_LIVE
    operator T&& ()  
  {
    return hostrpc::cxx::move(payload);
  }

  

  HOSTRPC_CALL_ON_DEAD  ~maybe() {}
  
 private:
 T payload;
 bool valid;


 
};

MODULE(maybe)
{
  using maybe_t = maybe<uint32_t>;
  TEST("happy path, false")
    {
      
      maybe_t  i (12, false);
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
      maybe_t i(12, true);
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
      // maybe_t i(12, false);
      // maybe_t v(24, true);
    }
  
  TEST("fail to check it")
    {
      // maybe_t i(12, true);
      // uint32_t v = i; (void)v;
    }

  TEST("check and don't use")
    {
      maybe_t i(12, false);
      i.unknown();
      if (i)
        {
          i.unconsumed();
        }
      i.unknown();
      if (i) {uint32_t d = i; (void)d; }
      i.consumed();
    }

  TEST("converting to bool doesn't change state")
    {
      maybe_t i(12, false);
      bool v = i;
      i.unknown();
      if (v)
        {
          i.unknown();
        }
      i.unknown();

      if (i) {uint32_t d = i; (void)d; }
      i.consumed();
    }
  
}

MAIN_MODULE() { DEPENDS(create_and_immediately_destroy); DEPENDS(maybe); }
