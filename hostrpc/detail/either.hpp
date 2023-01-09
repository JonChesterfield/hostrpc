#ifndef HOSTRPC_DETAIL_EITHER_HPP_INCLUDED
#define HOSTRPC_DETAIL_EITHER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "cxx.hpp"
#include "maybe.hpp"
#include "typestate.hpp"

namespace hostrpc
{
template <typename TrueTy, typename FalseTy>
struct HOSTRPC_CONSUMABLE_CLASS either;

#if HOSTRPC_USE_TYPESTATE
namespace cxx
{
// Declare, to put them in the right namespace
// Pretty sure none of these should be constexpr
template <typename TrueTy, typename FalseTy>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy> move(
    either<TrueTy, FalseTy> &&x HOSTRPC_CONSUMED_ARG);
template <typename TrueTy, typename FalseTy>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy> move(
    either<TrueTy, FalseTy> &x HOSTRPC_CONSUMED_ARG);
}  // namespace cxx
#endif

// Result is a type that will raise -Wconsumable errors if not 'used'
// Pattern is to return an instance of this from a function and branch
// on operator bool(). Use on_true/on_false in the respective branches
// to retrieve a TrueTy or FalseTy instance.

template <typename TrueTy, typename FalseTy>
struct HOSTRPC_CONSUMABLE_CLASS either
{
  static_assert(!cxx::is_same<TrueTy, FalseTy>(), "");

  static_assert(cxx::is_same<typename TrueTy::UnderlyingType,
                             typename FalseTy::UnderlyingType>(),
                "");

  using UnderlyingType = typename TrueTy::UnderlyingType;

  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  explicit operator bool() { return is_state<true>(); }

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  hostrpc::maybe<TrueTy> on_true()
  {
    if (*this)
      {
        TrueTy tmp = TrueTy::reconstitute({}, payload);
        tmp.unconsumed();
        return tmp;
      }
    else
      {
        return {};
      }
  }

  // Internal structuring is a little strange. It ensures that only the TrueTy is
  // ever used to create an instance, and likewise that only the TrueTy is used
  // to reconstitute the stored value. Inverting the instance at various points
  // means that PortUnderlyingAccess permissions are only granted to disassemble
  // and reconstruct that type. Provided either reliably distinguishes left from
  // right internally, this should stop it accidentally changing the type of the
  // stored object during the disassemble/reconstitute path.

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  hostrpc::maybe<FalseTy> on_false()
  {
    either<FalseTy, TrueTy> tmp = *this;
    return tmp.on_true();
  }

  template <typename CallbackReturn, typename OnTrue, typename OnFalse>
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  CallbackReturn visit(OnTrue on_true, OnFalse on_false)
  {
    if (*this)
      {
        return on_true(TrueTy::reconstitute({}, payload));
      }
    else
      {
        // return on_false(FalseTy::reconstitute({}, payload));
        either<FalseTy, TrueTy> tmp = *this;
        return tmp.template visit<CallbackReturn>(on_false, on_true);
      }
  }

  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE ~either() {}

  // Useful for checking assumptions
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

  HOSTRPC_CREATED_RES
  HOSTRPC_ANNOTATE
  static either Left(HOSTRPC_CONSUMED_ARG TrueTy &value)
  {
    either r{value.disassemble({}), true};
    r.unconsumed();
    return cxx::move(r);
  }

  HOSTRPC_CREATED_RES
  HOSTRPC_ANNOTATE
  static either Right(HOSTRPC_CONSUMED_ARG FalseTy &value)
  {
    using inverse = either<FalseTy, TrueTy>;
    return inverse::Left(value);
  }

  HOSTRPC_ANNOTATE
  static either Left(TrueTy &&value) { return Left(value); }
  HOSTRPC_ANNOTATE
  static either Right(FalseTy &&value) { return Right(value); }

  // Swap branches, consuming current instance in the process
  // Friend of self for this purpose
  friend either<FalseTy, TrueTy>;
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator either<FalseTy, TrueTy>() { return {payload, !state}; }

  // Either is movable
  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD
  either(either &&other HOSTRPC_CONSUMED_ARG)
      : payload(other.payload), state(other.state)
  {
    other.kill();
    def();
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD either &operator=(either &&other HOSTRPC_CONSUMED_ARG)
  {
    payload = other.payload;
    state = other.state;
    other.kill();
    def();
    return *this;
  }

 private:
  HOSTRPC_CREATED_RES
  HOSTRPC_ANNOTATE
  either(UnderlyingType payload, bool state) : payload(payload), state(state)
  {
    unknown();
  }

  template <bool State>
  HOSTRPC_ANNOTATE bool is_state()
  {
    return state == State;
  }

  UnderlyingType payload;
  bool state;  // can't be const and keep move assignment

#if HOSTRPC_USE_TYPESTATE
  // Declare move hooks as friends
  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy>
  cxx::move(either<TrueTy, FalseTy> &&x HOSTRPC_CONSUMED_ARG);

  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy>
  cxx::move(either<TrueTy, FalseTy> &x HOSTRPC_CONSUMED_ARG);
#endif

  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &&x HOSTRPC_CONSUMED_ARG)
  {
    UnderlyingType v = x.payload;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &x HOSTRPC_CONSUMED_ARG)
  {
    UnderlyingType v = x.payload;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  // Copying or moving maybe doesn't work very intuitively, moving either should
  // be OK
  either(const either &other) = delete;
  either &operator=(const either &other) = delete;
};

#if HOSTRPC_USE_TYPESTATE
namespace cxx
{
template <typename TrueTy, typename FalseTy>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy> move(
    either<TrueTy, FalseTy> &&x HOSTRPC_CONSUMED_ARG)
{
  return either<TrueTy, FalseTy>::recreate(x);
}

template <typename TrueTy, typename FalseTy>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy> move(
    either<TrueTy, FalseTy> &x HOSTRPC_CONSUMED_ARG)
{
  return either<TrueTy, FalseTy>::recreate(x);
}
}  // namespace cxx
#endif

}  // namespace hostrpc

#endif
