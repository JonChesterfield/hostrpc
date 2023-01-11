#ifndef HOSTRPC_DETAIL_EITHER_HPP_INCLUDED
#define HOSTRPC_DETAIL_EITHER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "cxx.hpp"
#include "maybe.hpp"
#include "tuple.hpp"
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

template <typename TrueTy, typename FalseTy>
struct HOSTRPC_CONSUMABLE_CLASS either
{
  static_assert(cxx::is_same<typename TrueTy::UnderlyingType,
                             typename FalseTy::UnderlyingType>(),
                "Simplifies implementation vs. using a union");

  using SelfType = either<TrueTy, FalseTy>;
  using InverseType = either<FalseTy, TrueTy>;
  using maybe = hostrpc::maybe<SelfType>;

  using ContainedType = typename TrueTy::UnderlyingType;
  using UnderlyingType = cxx::tuple<ContainedType, bool>;

 private:
  friend InverseType;
  ContainedType payload;
  bool state;  // can't be const and keep move assignment

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  TrueTy retrieve() { return TrueTy::reconstitute({}, payload); }

  template <typename Op>
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE
      HOSTRPC_SET_TYPESTATE(consumed) auto propagate(Op op)
          -> decltype(cxx::declval<Op>()(cxx::declval<TrueTy &&>()))
  {
    return op(retrieve());
  }

 public:
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
    return InverseType::Left(value);
  }

  HOSTRPC_ANNOTATE
  static either Left(TrueTy &&value) { return Left(value); }
  HOSTRPC_ANNOTATE
  static either Right(FalseTy &&value) { return Right(value); }

  // This should probably return an enum which is switched'ed on instead
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  explicit operator bool() { return is_state<true>(); }

  template <typename Op>
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE HOSTRPC_RETURN_UNKNOWN
      HOSTRPC_ANNOTATE hostrpc::maybe<TrueTy> left(Op op)
  {
    static_assert(cxx::is_same<void, decltype(cxx::declval<Op>()(
                                         cxx::declval<FalseTy &&>()))>(),
                  "");
    if (*this)
      {
        return retrieve();
      }
    else
      {
        invert().template propagate<Op>(cxx::forward<Op>(op));
        return {};
      }
  }

  template <typename Op>
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE HOSTRPC_RETURN_UNKNOWN
      HOSTRPC_ANNOTATE hostrpc::maybe<FalseTy> right(Op op)
  {
    static_assert(
        cxx::is_same<void,
                     decltype(cxx::declval<Op>()(cxx::declval<TrueTy &&>()))>(),
        "");
    return invert().left(cxx::forward<Op>(op));
  }

  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_ANNOTATE
  TrueTy left_and_right()
  {
    // When types match, can unconditionally retrieve
    static_assert(cxx::is_same<TrueTy, FalseTy>(), "");
    return retrieve();
  }

  template <typename CallbackReturn, typename OnTrue, typename OnFalse>
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE CallbackReturn
      visit(OnTrue on_true, OnFalse on_false)
  {
    if (*this)
      {
        return propagate<OnTrue>(cxx::forward<OnTrue>(on_true));
      }
    else
      {
        return invert().template visit<CallbackReturn>(on_false, on_true);
      }
  }

  template <typename OnTrue, typename OnFalse>
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE
      auto foreach (OnTrue on_true, OnFalse on_false)
  {
    using TrueResTy =
        decltype(cxx::declval<OnTrue>()(cxx::declval<TrueTy &&>()));
    using FalseResTy =
        decltype(cxx::declval<OnFalse>()(cxx::declval<FalseTy &&>()));

    using ResultTy = either<TrueResTy, FalseResTy>;

    if (*this)
      {
        return ResultTy::Left(propagate<OnTrue>(cxx::forward<OnTrue>(on_true)));
      }
    else
      {
        return invert()
            .template foreach<OnFalse, OnTrue>(on_false, on_true)
            .invert();
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
  InverseType invert() { return {payload, !state}; }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator InverseType() { return invert(); }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_RETURN_UNKNOWN
  operator maybe()
  {
    UnderlyingType u = {payload, state};
    return {typename maybe::Key{}, u};
  }

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
  either(ContainedType payload, bool state) : payload(payload), state(state)
  {
    unknown();
  }

  template <bool State>
  HOSTRPC_ANNOTATE bool is_state()
  {
    return state == State;
  }

  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

#if HOSTRPC_USE_TYPESTATE
  // Declare move hooks as friends
  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy>
  cxx::move(either<TrueTy, FalseTy> &&x HOSTRPC_CONSUMED_ARG);

  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr either<TrueTy, FalseTy>
  cxx::move(either<TrueTy, FalseTy> &x HOSTRPC_CONSUMED_ARG);
#endif

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &&x HOSTRPC_CONSUMED_ARG)
  {
    ContainedType v = x.payload;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  HOSTRPC_ANNOTATE static either HOSTRPC_CREATED_RES
  recreate(either &x HOSTRPC_CONSUMED_ARG)
  {
    ContainedType v = x.payload;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  either(const either &other) = delete;
  either &operator=(const either &other) = delete;

  // Implement the PortUnderlyingAccess / disassemble / reconstitute hook here
  // to allow constructing SelfType::maybe or either of either instances.
  // Private as the only classes that can access this are also friends of this
  // class.
  class PortUnderlyingAccess
  {
   private:
    template <typename L, typename R>
    friend struct either;
    friend hostrpc::maybe<SelfType>;
    HOSTRPC_ANNOTATE PortUnderlyingAccess() {}
    HOSTRPC_ANNOTATE PortUnderlyingAccess(PortUnderlyingAccess const &) {}
  };

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  UnderlyingType disassemble(PortUnderlyingAccess) { return {payload, state}; }

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES static either<TrueTy, FalseTy>
  reconstitute(PortUnderlyingAccess, UnderlyingType value)
  {
    return {value.template get<0>(), value.template get<1>()};
  }
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
