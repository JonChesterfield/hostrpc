#ifndef HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED
#define HOSTRPC_DETAIL_MAYBE_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "typestate.hpp"

namespace hostrpc
{
// Stores a T and constructs a U from it on request, if available
// Non-default construction is only permitted by U. This makes the
// class less usable in exchange for preventing out of thin air
// construction of U via this class.
template <typename T, typename U>
struct HOSTRPC_CONSUMABLE_CLASS maybe
{
  // Warning: When returning an instance from a function, that
  // function also needs to be annotated with:
  // HOSTRPC_RETURN_UNKNOWN
  // Otherwise the default is unconsumed which will usually error on the bool()

  // Starts out with unknown type state

  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  maybe() : valid(false) { unknown(); }

 private:
  friend U;

  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  maybe(T payload) : payload(static_cast<T &&>(payload)), valid(true)
  {
    unknown();
  }

 public:
  // Branch on the value, the true side will be 'unconsumed'
  HOSTRPC_CALL_ON_UNKNOWN
  HOSTRPC_ANNOTATE
  explicit operator bool() HOSTRPC_TEST_TYPESTATE(unconsumed) { return valid; }

  // When in the true branch, extract T exactly once
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  U value() { return payload; }

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
