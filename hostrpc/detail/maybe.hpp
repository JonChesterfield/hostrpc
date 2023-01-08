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
template <typename T>
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
  
  // Branch on the value, the true side will be 'unconsumed'
  HOSTRPC_CALL_ON_UNKNOWN
  HOSTRPC_ANNOTATE
  explicit operator bool() HOSTRPC_TEST_TYPESTATE(unconsumed) { return valid; }

  // When in the true branch, extract T exactly once
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  T value() { return T::reconstitute({}, payload); }
  
  // Errors if the above pattern is not followed
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE ~maybe() {}

  // Useful for checking assumptions
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}


  // Used to implement operator maybe() from T
  //
  // The idea here was to pass maybe a constructed port instance and have it
  // disassemble and reconstitute it. Unfortunately that has proven
  // challenging to get past the consumed annotation, which really prefers
  // user defined conversion to maybe() defined on the port types.
  // Can therefore either have a private constructor and declare T a friend
  // of maybe, which works, or do this slightly more contrained thing where
  // T is the only type that can construct a key instance with which to call
  // the public constructor
  // Might be better to require a default constructor for T::UnderlyingType
  // and make that field const, then give T friend access to this class on the
  // basis that it can't change any of it anyway.
 class Key
 {
 private:
   // T is allowed access to the constructor but nothing else
   friend T;
   HOSTRPC_ANNOTATE  Key() {}
   HOSTRPC_ANNOTATE  Key(Key const&) {}
 };
  
  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  maybe(Key, typename  T::UnderlyingType v) : payload(v), valid(true) {}

 private:
  
  typename  T::UnderlyingType payload;
  const bool valid;

  // Copying or moving these types doesn't work very intuitively
  maybe(const maybe &other) = delete;
  maybe(maybe &&other) = delete;
  maybe &operator=(const maybe &other) = delete;
  maybe &operator=(maybe &&other) = delete;
};
}  // namespace hostrpc

#endif
