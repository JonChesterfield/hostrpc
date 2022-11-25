#ifndef TYPED_PORT_T_HPP_INCLUDED
#define TYPED_PORT_T_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "typestate.hpp"

#include "cxx.hpp"
#include "either.hpp"
#include "maybe.hpp"
#include "tuple.hpp"  // may be better to make maybe construction variadic

namespace hostrpc
{
// This is essentially a uint32_t used to index into the arrays within the state
// machine. It tracks the inbox and outbox state in the template parameters
// and detects use-after-move and missing calls to close_port using the
// clang typestate machinery.

template <typename Friend, unsigned I, unsigned O>
class typed_port_impl_t;

template <typename Friend, unsigned S>
class partial_port_impl_t;

// Typed port knows the exact state of inbox and outbox at a given point in time
// Partial port knows whether inbox == outbox, called Stable / S here, and
// tracks what the the inbox/outbox state is using a runtime boolean.
// These are exactly bidirectionally convertible, at least for a compile time
// bool Document the mapping in static_assert checked form here.

namespace traits
{
template <typename Friend, typename T>
struct typed_to_partial_trait;

template <typename Friend, typename T, bool state>
struct partial_to_typed_trait;

// <0, 0> -> S == 1, state == false
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 0, 0>>
{
  using type = partial_port_impl_t<Friend, 1>;
  static constexpr bool state = false;
};

// <0, 1> -> S == 0, state == true
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 0, 1>>
{
  using type = partial_port_impl_t<Friend, 0>;
  static constexpr bool state = true;
};

// <1, 1> -> S == 1, state == true
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 1, 1>>
{
  using type = partial_port_impl_t<Friend, 1>;
  static constexpr bool state = true;
};

// <1, 0> -> S == 0, state == false
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 1, 0>>
{
  using type = partial_port_impl_t<Friend, 0>;
  static constexpr bool state = false;
};

// equal and outbox true
template <typename Friend>
struct partial_to_typed_trait<Friend, partial_port_impl_t<Friend, 1>, true>
{
  using type = typed_port_impl_t<Friend, 1, 1>;
};

// equal and outbox false
template <typename Friend>
struct partial_to_typed_trait<Friend, partial_port_impl_t<Friend, 1>, false>
{
  using type = typed_port_impl_t<Friend, 0, 0>;
};

// differ and outbox true
template <typename Friend>
struct partial_to_typed_trait<Friend, partial_port_impl_t<Friend, 0>, true>
{
  using type = typed_port_impl_t<Friend, 0, 1>;
};

// differ and outbox false
template <typename Friend>
struct partial_to_typed_trait<Friend, partial_port_impl_t<Friend, 0>, false>
{
  using type = typed_port_impl_t<Friend, 1, 0>;
};

template <class T, class U>
struct is_same : cxx::false_type
{
};

template <class T>
struct is_same<T, T> : cxx::true_type
{
};

// Make a typed port type, convert to partial port and back
template <typename Friend, unsigned I, unsigned O>
struct check_from_typed
{
  using typed_port_t = typed_port_impl_t<Friend, I, O>;

  using partial_info = typed_to_partial_trait<Friend, typed_port_t>;

  using typed_info = partial_to_typed_trait<Friend, typename partial_info::type,
                                            partial_info::state>;

  static constexpr bool consistent()
  {
    return is_same<typed_port_t, typename typed_info::type>::value;
  }
};

// Make a partial port type, convert to typed port and back
template <typename Friend, unsigned S, bool state>
struct check_from_partial
{
  using partial_port_t = partial_port_impl_t<Friend, S>;

  using typed_info = partial_to_typed_trait<Friend, partial_port_t, state>;

  using partial_info =
      typed_to_partial_trait<Friend, typename typed_info::type>;

  static constexpr bool consistent()
  {
    return is_same<partial_port_t, typename partial_info::type>::value &&
           partial_info::state == state;
  }
};

// Check all four states of each type convert back and forth consistently
template <typename Friend>
constexpr bool traits_consistent()
{
  return check_from_typed<Friend, 0, 0>::consistent() &&
         check_from_typed<Friend, 0, 1>::consistent() &&
         check_from_typed<Friend, 1, 0>::consistent() &&
         check_from_typed<Friend, 1, 1>::consistent() &&
         check_from_partial<Friend, 0, false>::consistent() &&
         check_from_partial<Friend, 1, false>::consistent() &&
         check_from_partial<Friend, 0, true>::consistent() &&
         check_from_partial<Friend, 1, true>::consistent();
}
}  // namespace traits

#if HOSTRPC_USE_TYPESTATE
// It's going to have it's own implementation of cxx::move
namespace cxx
{
template <typename F, unsigned I, unsigned O>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr typed_port_impl_t<F, I, O> move(
    typed_port_impl_t<F, I, O> &&x HOSTRPC_CONSUMED_ARG);
template <typename F, unsigned I, unsigned O>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr typed_port_impl_t<F, I, O> move(
    typed_port_impl_t<F, I, O> &x HOSTRPC_CONSUMED_ARG);

template <typename F, unsigned S>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr partial_port_impl_t<F, S> move(
    partial_port_impl_t<F, S> &&x HOSTRPC_CONSUMED_ARG);
template <typename F, unsigned S>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr partial_port_impl_t<F, S> move(
    partial_port_impl_t<F, S> &x HOSTRPC_CONSUMED_ARG);

}  // namespace cxx
#endif

template <typename Friend, unsigned I, unsigned O>
class HOSTRPC_CONSUMABLE_CLASS typed_port_impl_t
{
 private:
  using SelfType = typed_port_impl_t<Friend, I, O>;
  friend Friend;  // the state machine
  uint32_t value;

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES typed_port_impl_t(uint32_t v) : value(v)
  {
    static_assert((I <= 1) && (O <= 1), "");
  }

#if HOSTRPC_USE_TYPESTATE
  // so that cxx::move keeps track of the typestate
  friend HOSTRPC_ANNOTATE
      HOSTRPC_CREATED_RES constexpr typed_port_impl_t<Friend, I, O>
      cxx::move(typed_port_impl_t<Friend, I, O> &&x HOSTRPC_CONSUMED_ARG);

  friend HOSTRPC_ANNOTATE
      HOSTRPC_CREATED_RES constexpr typed_port_impl_t<Friend, I, O>
      cxx::move(typed_port_impl_t<Friend, I, O> &x HOSTRPC_CONSUMED_ARG);
#endif

  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

 public:
  static constexpr unsigned InboxState = I;
  static constexpr unsigned OutboxState = O;

  // can convert it back to a uint32_t for indexing into structures
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE operator uint32_t() const
  {
    return value;
  }

  // non-default maybe can only be constructed by the second template parameter,
  // i.e. by this class. The only method that does so is operator that consumes
  // the port. Thus this instance can be converted to a maybe and then
  // retrieved.
  using maybe = hostrpc::maybe<uint32_t, SelfType>;
  friend maybe;

  template <bool InboxSet, bool OutboxSet>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN static maybe make(uint32_t v)
  {
    constexpr unsigned ReqInbox = InboxSet ? 1 : 0;
    constexpr unsigned ReqOutbox = OutboxSet ? 1 : 0;
    if (I == ReqInbox && O == ReqOutbox)
      {
        return {v};
      }
    else
      {
        return {};
      }
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_RETURN_UNKNOWN
  operator maybe()
  {
    uint32_t v = *this;
    kill();
    return {v};
  }

  // either_builder can only be constructed by the first template parameter,
  // i.e. by this class. The only method that does so consumes the port. either
  // is only constructed from either_builder, which consumes the builder.
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator hostrpc::either_builder<SelfType, typed_port_impl_t<Friend, I, !O>,
                                   uint32_t>()
  {
    uint32_t v = *this;
    kill();
    return {v};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator hostrpc::either_builder<SelfType, typed_port_impl_t<Friend, !I, O>,
                                   uint32_t>()
  {
    uint32_t v = *this;
    kill();
    return {v};
  }

  
  // if either is constructed by normal(), the ==true path is live and will
  // return (a maybe that returns) SelfType.
  // if either is cosntructed by invert(), the ==false path is live and will
  // return (a maybe that returns) SelfType.
  // The dynamically dead path will call the other constructor, which is a
  // friend here to allow the dead path to typecheck.
  
  // construction with outbox changed
  friend hostrpc::either<SelfType, typed_port_impl_t<Friend, I, !O>, uint32_t>;
  friend hostrpc::either<typed_port_impl_t<Friend, I, !O>, SelfType, uint32_t>;

  // construction with inbox changed
  friend hostrpc::either<SelfType, typed_port_impl_t<Friend, !I, O>, uint32_t>;
  friend hostrpc::either<typed_port_impl_t<Friend, !I, O>, SelfType, uint32_t>;
  
  // move construct and assign are available
  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD
  typed_port_impl_t(typed_port_impl_t &&other HOSTRPC_CONSUMED_ARG)
      : value(other.value)
  {
    other.kill();
    def();
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD typed_port_impl_t &operator=(
      typed_port_impl_t &&other HOSTRPC_CONSUMED_ARG)
  {
    value = other.value;
    other.kill();
    def();
    return *this;
  }

  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_DEAD ~typed_port_impl_t() {}

  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

  // leaves value uninitialised, uses of the value are caught
  // by the typestate annotations
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_CONSUMED typed_port_impl_t() {}

 private:
  HOSTRPC_ANNOTATE static typed_port_impl_t HOSTRPC_CREATED_RES
  recreate(typed_port_impl_t &&x HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = x;
    x.kill();
    return v;
  }

  HOSTRPC_ANNOTATE static typed_port_impl_t HOSTRPC_CREATED_RES
  recreate(typed_port_impl_t &x HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = x;
    x.kill();
    return v;
  }

  typed_port_impl_t(const typed_port_impl_t &) = delete;
  typed_port_impl_t &operator=(const typed_port_impl_t &) = delete;
};

template <typename Friend, unsigned S>
class HOSTRPC_CONSUMABLE_CLASS partial_port_impl_t
{
 private:
  using SelfType = partial_port_impl_t<Friend, S>;
  friend Friend;  // the state machine
  uint32_t value;
  bool state;

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES partial_port_impl_t(uint32_t v,
                                                           bool state)
      : value(v), state(state)
  {
    static_assert((S == 0) || (S == 1), "");
  }

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES
  partial_port_impl_t(cxx::tuple<uint32_t, bool> tup)
      : value(tup.get<0>()), state(tup.get<1>())
  {
  }

#if HOSTRPC_USE_TYPESTATE
  // so that cxx::move keeps track of the typestate
  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES partial_port_impl_t<Friend, S>
  cxx::move(partial_port_impl_t<Friend, S> &&x HOSTRPC_CONSUMED_ARG);

  friend HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES partial_port_impl_t<Friend, S>
  cxx::move(partial_port_impl_t<Friend, S> &x HOSTRPC_CONSUMED_ARG);
#endif

  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(consumed) void kill() const {}
  HOSTRPC_ANNOTATE HOSTRPC_SET_TYPESTATE(unconsumed) void def() const {}

 public:
  // can convert it back to a uint32_t for indexing into structures
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE operator uint32_t() const
  {
    return value;
  }

  using maybe = hostrpc::maybe<cxx::tuple<uint32_t, bool>, SelfType>;
  friend maybe;

  template <bool InboxSet, bool OutboxSet>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN static maybe make(uint32_t v)
  {
    constexpr unsigned ReqSame = (InboxSet == OutboxSet) ? 1 : 0;
    if (ReqSame == S)
      {
        cxx::tuple<uint32_t, bool> tup = {v, OutboxSet};
        return {tup};
      }
    else
      {
        return {};
      }
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_RETURN_UNKNOWN
  operator maybe()
  {
    cxx::tuple<uint32_t, bool> tup = {value, state};
    kill();
    return {tup};
  }

  // move construct and assign are available
  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD
  partial_port_impl_t(partial_port_impl_t &&other HOSTRPC_CONSUMED_ARG)
      : value(other.value), state(other.state)
  {
    other.kill();
    def();
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CREATED_RES
  HOSTRPC_CALL_ON_DEAD partial_port_impl_t &operator=(
      partial_port_impl_t &&other HOSTRPC_CONSUMED_ARG)
  {
    value = other.value;
    state = other.state;
    other.kill();
    def();
    return *this;
  }

  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_DEAD ~partial_port_impl_t() {}

  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

  // leaves value uninitialised, uses of the value are caught
  // by the typestate annotations
  // else error, default must explicitly initialise const member
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_CONSUMED partial_port_impl_t() : state(false)
  {
  }

  template <bool OutboxState>
  HOSTRPC_ANNOTATE bool outbox()
  {
    return state == OutboxState;
  }

  template <bool InboxState>
  HOSTRPC_ANNOTATE bool inbox()
  {
    return (S == 1) ? outbox() : !outbox();
  }

 private:
  HOSTRPC_ANNOTATE static partial_port_impl_t HOSTRPC_CREATED_RES
  recreate(partial_port_impl_t &&x HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = x.value;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  HOSTRPC_ANNOTATE static partial_port_impl_t HOSTRPC_CREATED_RES
  recreate(partial_port_impl_t &x HOSTRPC_CONSUMED_ARG)
  {
    uint32_t v = x.value;
    bool s = x.state;
    x.kill();
    return {v, s};
  }

  partial_port_impl_t(const partial_port_impl_t &) = delete;
  partial_port_impl_t &operator=(const partial_port_impl_t &) = delete;
};

#if HOSTRPC_USE_TYPESTATE
namespace cxx
{
template <typename F, unsigned I, unsigned O>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr typed_port_impl_t<F, I, O> move(
    typed_port_impl_t<F, I, O> &&x HOSTRPC_CONSUMED_ARG)
{
  return typed_port_impl_t<F, I, O>::recreate(x);
}

template <typename F, unsigned I, unsigned O>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr typed_port_impl_t<F, I, O> move(
    typed_port_impl_t<F, I, O> &x HOSTRPC_CONSUMED_ARG)
{
  return typed_port_impl_t<F, I, O>::recreate(x);
}

template <typename F, unsigned S>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr partial_port_impl_t<F, S> move(
    partial_port_impl_t<F, S> &&x HOSTRPC_CONSUMED_ARG)
{
  return partial_port_impl_t<F, S>::recreate(x);
}

template <typename F, unsigned S>
HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr partial_port_impl_t<F, S> move(
    partial_port_impl_t<F, S> &x HOSTRPC_CONSUMED_ARG)
{
  return partial_port_impl_t<F, S>::recreate(x);
}

}  // namespace cxx
#endif

}  // namespace hostrpc

#endif
