#ifndef HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED
#define HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "typestate.hpp"

namespace hostrpc
{

// By default stores a T and returns the same type, but if desired
// can store a different type and construct it on request
template <typename T, typename U = T>
struct HOSTRPC_CONSUMABLE_CLASS maybe
{
  // Warning: When returning an instance from a function, that
  // function also needs to be annotated with:
  // HOSTRPC_RETURN_UNKNOWN

  // Starts out with unknown type state

  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  maybe() : valid(false)
  {
    unknown();
  }

  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  maybe(T payload)
      : payload(static_cast<T &&>(payload)), valid(true)
  {
    unknown();
  }

  // Branch on the value, the true side will be 'unconsumed'
  HOSTRPC_CALL_ON_UNKNOWN
  HOSTRPC_ANNOTATE
  explicit operator bool() HOSTRPC_TEST_TYPESTATE(unconsumed) { return valid; }

  // When in the true branch, extract T exactly once
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  U value() { return static_cast<T &&>(payload); }

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  operator U() { return value(); }
  
  // Errors if the above pattern is not followed
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE ~maybe() {}

  // Useful for checking assumptions
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

 private:
  T payload;
  const bool valid;

  // Copying or moving these types doesn't work very intuitively
  maybe(const maybe &other) = delete;
  maybe(maybe &&other) = delete;
  maybe &operator=(const maybe &other) = delete;
  maybe &operator=(maybe &&other) = delete;
};
}  // namespace hostrpc

#endif
