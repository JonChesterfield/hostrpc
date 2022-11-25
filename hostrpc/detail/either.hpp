#ifndef HOSTRPC_DETAIL_EITHER_HPP_INCLUDED
#define HOSTRPC_DETAIL_EITHER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "maybe.hpp"
#include "typestate.hpp"

namespace hostrpc
{

template <typename TrueTy, typename FalseTy, typename From>
struct HOSTRPC_CONSUMABLE_CLASS either;

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

  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  explicit operator bool() { return is_state<true>(); }

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_RETURN_UNKNOWN
  HOSTRPC_ANNOTATE
  hostrpc::maybe<From, TrueTy> on_true()
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
  hostrpc::maybe<From, FalseTy> on_false()
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
  const bool state;

  // Copying or moving these types doesn't work very intuitively
  either(const either &other) = delete;
  either(either &&other) = delete;
  either &operator=(const either &other) = delete;
  either &operator=(either &&other) = delete;
};

}  // namespace hostrpc

#endif
