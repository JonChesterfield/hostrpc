#ifndef HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED
#define HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED

#include "typestate.hpp"

namespace hostrpc {

template <typename T>
struct HOSTRPC_CONSUMABLE_CLASS maybe
{
  // Warning: When returning an instance from a function, that
  // function also needs to be annotated with:
  // HOSTRPC_RETURN_UNKNOWN

  // Starts out with unknown type state
 #if 0
  HOSTRPC_RETURN_UNKNOWN
  maybe(T payload, bool valid)
    : payload(static_cast<T&&>(payload)), valid(valid)
  {
    unknown();
  }
  #endif

  HOSTRPC_RETURN_UNKNOWN
  maybe(T&& payload, bool valid)
    : payload(static_cast<T&&>(payload)), valid(valid)
  {
    unknown();
  }
  

  // Branch on the value, the true side will be 'unconsumed'
  HOSTRPC_CALL_ON_UNKNOWN
  explicit operator bool() HOSTRPC_TEST_TYPESTATE(unconsumed)
  {
    return valid;
  }
  
  // When in the true branch, extract T exactly once
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
    operator T &&() { return static_cast<T&&>(payload); }

  // Errors if the above pattern is not followed
  HOSTRPC_CALL_ON_DEAD ~maybe() {}

  // Useful for checking assumptions
  HOSTRPC_CALL_ON_DEAD void consumed() const {}
  HOSTRPC_CALL_ON_LIVE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN void unknown() const {}


 private:
  T payload;
  bool valid;

  // Copying or moving these types doesn't work very intuitively
  maybe(const maybe &other) = delete;
  maybe(maybe &&other) = delete;
  maybe &operator=(const maybe &other) = delete;
  maybe &operator=(maybe &&other) = delete;
};
}


#endif
