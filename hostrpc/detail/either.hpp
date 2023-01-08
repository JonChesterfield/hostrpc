#ifndef HOSTRPC_DETAIL_EITHER_HPP_INCLUDED
#define HOSTRPC_DETAIL_EITHER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "maybe.hpp"
#include "typestate.hpp"

namespace hostrpc
{
template <typename TrueTy, typename FalseTy, typename From>
struct HOSTRPC_CONSUMABLE_CLASS either;

#if HOSTRPC_USE_TYPESTATE
namespace cxx
{
// Declare, to put them in the right namespace
// Pretty sure none of these should be constexpr
template <typename TrueTy, typename FalseTy, typename From>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
move(either<TrueTy, FalseTy, From> &&x HOSTRPC_CONSUMED_ARG);
template <typename TrueTy, typename FalseTy, typename From>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
move(either<TrueTy, FalseTy, From> &x HOSTRPC_CONSUMED_ARG);
}  // namespace cxx
#endif

template <typename TrueTy, typename FalseTy, typename From>
struct HOSTRPC_CONSUMABLE_CLASS either_builder
{
  // Constructed by the TrueTy instance. Consumed by either normal() or
  // invert(). This keeps the state field of either consistent with which type
  // constructed it.

  friend TrueTy;

  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CREATED_RES
  HOSTRPC_ANNOTATE
  either<TrueTy, FalseTy, From> normal() { return {payload, true}; }

  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CREATED_RES
  HOSTRPC_ANNOTATE
  either<FalseTy, TrueTy, From> invert() { return {payload, false}; }

  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_DEAD ~either_builder() {}
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

 private:
  From payload;
  HOSTRPC_ANNOTATE
  either_builder(From payload) : payload(payload) {}
};

// Construct from a single-use instance of either_builder.
// Result is a type that will raise -Wconsumable errors if not 'used'
// Pattern is to return an instance of this from a function and branch
// on operator bool(). Use on_true/on_false in the respective branches
// to retrieve a TrueTy or FalseTy instance constructed from the stored From.

template <typename TrueTy, typename FalseTy, typename From>
struct HOSTRPC_CONSUMABLE_CLASS either
{
  friend either_builder<TrueTy, FalseTy, From>;
  friend either_builder<FalseTy, TrueTy, From>;
  friend either<FalseTy, TrueTy, From>;

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
        TrueTy tmp(payload);
        return tmp;
      }
    else
      {
        return {};
      }
  }

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  hostrpc::maybe<FalseTy> on_false()
  {
    if (!*this)
      {
        FalseTy tmp(payload);
        return tmp;
      }
    else
      {
        return {};
      }
  }

  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE ~either() {}

  // Useful for checking assumptions
  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

  // Swap branches, consuming current instance in the process
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator either<FalseTy, TrueTy, From>() { return {payload, !state}; }

  // Going to try allowing moving either
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
  either(From payload, bool state) : payload(payload), state(state)
  {
    unknown();
  }

  template <bool State>
  HOSTRPC_ANNOTATE bool is_state()
  {
    return state == State;
  }

  From payload;
  bool state;  // can't be const and keep move assignment

#if HOSTRPC_USE_TYPESTATE
  // Declare move hooks as friends
  friend HOSTRPC_ANNOTATE
      HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
      cxx::move(either<TrueTy, FalseTy, From> &&x HOSTRPC_CONSUMED_ARG);

  friend HOSTRPC_ANNOTATE
      HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
      cxx::move(either<TrueTy, FalseTy, From> &x HOSTRPC_CONSUMED_ARG);
#endif

  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &&x HOSTRPC_CONSUMED_ARG)
  {
    From v = x.payload;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &x HOSTRPC_CONSUMED_ARG)
  {
    From v = x.payload;
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
template <typename TrueTy, typename FalseTy, typename From>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
move(either<TrueTy, FalseTy, From> &&x HOSTRPC_CONSUMED_ARG)
{
  return either<TrueTy, FalseTy, From>::recreate(x);
}

template <typename TrueTy, typename FalseTy, typename From>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy, From>
move(either<TrueTy, FalseTy, From> &x HOSTRPC_CONSUMED_ARG)
{
  return either<TrueTy, FalseTy, From>::recreate(x);
}
}  // namespace cxx
#endif

}  // namespace hostrpc

#endif
