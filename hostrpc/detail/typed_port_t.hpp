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
  static constexpr bool state() { return false; }
};

// <0, 1> -> S == 0, state == true
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 0, 1>>
{
  using type = partial_port_impl_t<Friend, 0>;
  static constexpr bool state() { return true; }
};

// <1, 1> -> S == 1, state == true
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 1, 1>>
{
  using type = partial_port_impl_t<Friend, 1>;
  static constexpr bool state() { return true; }
};

// <1, 0> -> S == 0, state == false
template <typename Friend>
struct typed_to_partial_trait<Friend, typed_port_impl_t<Friend, 1, 0>>
{
  using type = partial_port_impl_t<Friend, 0>;
  static constexpr bool state() { return false; }
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

// Make a typed port type, convert to partial port and back
template <typename Friend, unsigned I, unsigned O>
struct check_from_typed
{
  using typed_port_t = typed_port_impl_t<Friend, I, O>;

  using partial_info = typed_to_partial_trait<Friend, typed_port_t>;

  using typed_info = partial_to_typed_trait<Friend, typename partial_info::type,
                                            partial_info::state()>;

  static constexpr bool consistent()
  {
    return cxx::is_same<typed_port_t, typename typed_info::type>() /*::value*/;
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
    return cxx::is_same<partial_port_t,
                        typename partial_info::type>() /*::value*/
           && partial_info::state() == state;
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

// Trivial ABI, __attribute__((trivial_abi)), would be a really good fit for
// this class. It can be passed / returned in an i32 register, all the deleted
// move constructors and so forth should burn out by codegen. With that type
// applied and the typestate stuff quite crudely hacked out this is indeed
// returned as an i32. I don't think it can be passed as i32 unless the
// cxx::move() requirement at call sites can be dropped, perhaps by annotating
// the state machine methods directly.

template <typename Friend, unsigned I, unsigned O>
class HOSTRPC_CONSUMABLE_CLASS typed_port_impl_t
{
 public:
  using UnderlyingType = uint32_t;
  using SelfType = typed_port_impl_t<Friend, I, O>;
  using maybe = hostrpc::maybe<SelfType>;

 private:
  UnderlyingType value;

  friend Friend;  // the state machine

  // The idea here is to track access permissions at finer grain than wholly trusted.
  // Ideally converting to/from the UnderlyingType would be limited to environments
  // which cannot also change the state, e.g. can't call invert_inbox.
  // Converting directly betweeen typed and partial port is probably a special case.

  class PortUnderlyingAccess
  {
   private:
    friend typed_port_impl_t<Friend, I, O>;
    friend partial_port_impl_t<Friend, I == O>;

    friend hostrpc::maybe<typed_port_impl_t<Friend, I, O>>;

    // either can only construct from an instance of its lhs or its rhs type
    // this restricts it to only construct (and disassemble) from the lhs
    // that should suffice to stop either from constructing ports with the
    // type annotation changed. Internally it handles this by inverting.
    // At least, it would also need to mishandle it's own state to go wrong.
    friend either<typed_port_impl_t<Friend, I, O>,
                  typed_port_impl_t<Friend, !I, O>>;

    friend either<typed_port_impl_t<Friend, I, O>,
                  typed_port_impl_t<Friend, I, !O>>;

    friend either<typed_port_impl_t<Friend, I, O>,
                  typed_port_impl_t<Friend, !I, !O>>;

    // Also type preserving to have lhs==rhs
    friend either<typed_port_impl_t<Friend, I, O>,
                  typed_port_impl_t<Friend, I, O>>;

    HOSTRPC_ANNOTATE PortUnderlyingAccess() {}
    HOSTRPC_ANNOTATE PortUnderlyingAccess(PortUnderlyingAccess const &) {}
  };

  class InboxPermission
  {
  private:
    friend typename Friend::inbox_t;
    friend Friend; // temporary
    HOSTRPC_ANNOTATE InboxPermission() {}
    HOSTRPC_ANNOTATE InboxPermission(InboxPermission const &) {}    
  };

  class OutboxPermission
  {
  private:
    friend typename Friend::outbox_t;
    HOSTRPC_ANNOTATE OutboxPermission() {}
    HOSTRPC_ANNOTATE OutboxPermission(OutboxPermission const &) {}    
  };
  
 public:
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  UnderlyingType disassemble(PortUnderlyingAccess) { return value; }

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES static SelfType reconstitute(
      PortUnderlyingAccess, UnderlyingType value)
  {
    return {value};
  }

 private:
  // Constructor is private. Permissions setup is fairly complicated.
  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES typed_port_impl_t(uint32_t v) : value(v)
  {
    static_assert((I <= 1) && (O <= 1), "");
  }

  template <bool InboxSet, bool OutboxSet>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN static maybe make(uint32_t v)
  {
    constexpr unsigned ReqInbox = InboxSet ? 1 : 0;
    constexpr unsigned ReqOutbox = OutboxSet ? 1 : 0;
    if (I == ReqInbox && O == ReqOutbox)
      {
        return {{}, v};
      }
    else
      {
        return {};
      }
  }

  // Trust instances of this type with inbox/outbox inverted but not both
  // to support invert inbox_state/outbox_state
  friend typed_port_impl_t<Friend, I, !O>;
  friend typed_port_impl_t<Friend, !I, O>;

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
  // can convert it back to a uint32_t for indexing into structures
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE operator uint32_t() const
  {
    return value;
  }

  // non-constexpr member functions to match partial_port_impl_t
  HOSTRPC_ANNOTATE bool outbox_state() const { return O; }
  HOSTRPC_ANNOTATE bool inbox_state() const { return I; }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_CREATED_RES
  HOSTRPC_SET_TYPESTATE(consumed)
  typed_port_impl_t<Friend, I, !O> invert_outbox(OutboxPermission)
  {
    UnderlyingType v = *this;
    kill();
    return {v};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_CREATED_RES
  HOSTRPC_SET_TYPESTATE(consumed)
  typed_port_impl_t<Friend, !I, O> invert_inbox(InboxPermission)
  {
    UnderlyingType v = *this;
    kill();
    return {v};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_RETURN_UNKNOWN
  operator maybe()
  {
    UnderlyingType u = value;
    return {typename maybe::Key{}, u};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator typename traits::typed_to_partial_trait<Friend, SelfType>::type();

  // move construct and assign are available
#if HOSTRPC_USE_TYPESTATE
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

  // leaves value uninitialised, uses of the value are caught
  // by the typestate annotations
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_CONSUMED typed_port_impl_t() {}
#else
  HOSTRPC_ANNOTATE typed_port_impl_t() = default;
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_DEAD ~typed_port_impl_t() = default;
  typed_port_impl_t(typed_port_impl_t &&other) = default;
  typed_port_impl_t &operator=(typed_port_impl_t &&other) = default;

#endif

  HOSTRPC_CALL_ON_DEAD HOSTRPC_ANNOTATE void consumed() const {}
  HOSTRPC_CALL_ON_LIVE HOSTRPC_ANNOTATE void unconsumed() const {}
  HOSTRPC_CALL_ON_UNKNOWN HOSTRPC_ANNOTATE void unknown() const {}

 private:
  HOSTRPC_ANNOTATE static typed_port_impl_t HOSTRPC_CREATED_RES
  recreate(typed_port_impl_t &&x HOSTRPC_CONSUMED_ARG)
  {
    UnderlyingType v = x;
    x.kill();
    return v;
  }

  HOSTRPC_ANNOTATE static typed_port_impl_t HOSTRPC_CREATED_RES
  recreate(typed_port_impl_t &x HOSTRPC_CONSUMED_ARG)
  {
    UnderlyingType v = x;
    x.kill();
    return v;
  }

#if HOSTRPC_USE_TYPESTATE
  typed_port_impl_t(const typed_port_impl_t &) = delete;
  typed_port_impl_t &operator=(const typed_port_impl_t &) = delete;
#else
  typed_port_impl_t(const typed_port_impl_t &) = default;
  typed_port_impl_t &operator=(const typed_port_impl_t &) = default;

#endif
};

template <typename Friend, unsigned S>
class HOSTRPC_CONSUMABLE_CLASS partial_port_impl_t
{
 public:
  using UnderlyingType = cxx::tuple<uint32_t, bool>;
  using SelfType = partial_port_impl_t<Friend, S>;
  using maybe = hostrpc::maybe<SelfType>;

 private:
  uint32_t value;
  bool state;

  friend Friend;  // the state machine

  class PortUnderlyingAccess
  {
   private:
    friend partial_port_impl_t<Friend, S>;
    friend typed_port_impl_t<Friend, 0, S ? 0 : 1>;
    friend typed_port_impl_t<Friend, 1, S ? 1 : 0>;

    friend hostrpc::maybe<partial_port_impl_t<Friend, S>>;

    friend either<partial_port_impl_t<Friend, S>,
                  partial_port_impl_t<Friend, !S>>;

    HOSTRPC_ANNOTATE PortUnderlyingAccess() {}
    HOSTRPC_ANNOTATE PortUnderlyingAccess(PortUnderlyingAccess const &) {}
  };

  class InboxPermission
  {
  private:
    friend typename Friend::inbox_t;
    HOSTRPC_ANNOTATE InboxPermission() {}
    HOSTRPC_ANNOTATE InboxPermission(InboxPermission const &) {}    
  };

  class OutboxPermission
  {
  private:
    friend typename Friend::outbox_t;
    HOSTRPC_ANNOTATE OutboxPermission() {}
    HOSTRPC_ANNOTATE OutboxPermission(OutboxPermission const &) {}    
  };
 
 public:
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  UnderlyingType disassemble(PortUnderlyingAccess) { return {value, state}; }

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES static SelfType reconstitute(
      PortUnderlyingAccess, UnderlyingType value)
  {
    return {value};
  }

 private:
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

  HOSTRPC_ANNOTATE bool outbox_state() const { return state; }
  HOSTRPC_ANNOTATE bool inbox_state() const
  {
    return (S == 1) ? outbox_state() : !outbox_state();
  }

  template <bool InboxSet, bool OutboxSet>
  HOSTRPC_ANNOTATE HOSTRPC_RETURN_UNKNOWN static maybe make(uint32_t v)
  {
    constexpr unsigned ReqSame = (InboxSet == OutboxSet) ? 1 : 0;
    if (ReqSame == S)
      {
        cxx::tuple<uint32_t, bool> tup = {v, OutboxSet};
        SelfType port(tup);
        return cxx::move(port);
      }
    else
      {
        return {};
      }
  }

  // For invert inbox/outbox
  friend partial_port_impl_t<Friend, !S>;

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_CREATED_RES
  HOSTRPC_SET_TYPESTATE(consumed)
  partial_port_impl_t<Friend, !S> invert_outbox(OutboxPermission)
  {
    // Inverts outbox and inverts S
    cxx::tuple<uint32_t, bool> tup = {value, !state};
    kill();
    return {tup};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_CREATED_RES
  HOSTRPC_SET_TYPESTATE(consumed)
  partial_port_impl_t<Friend, !S> invert_inbox(InboxPermission)
  {
    // No change to outbox, inverts S
    cxx::tuple<uint32_t, bool> tup = {value, state};
    kill();
    return {tup};
  }

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  HOSTRPC_RETURN_UNKNOWN
  operator maybe()
  {
    UnderlyingType u = {value, state};
    return {typename maybe::Key{}, u};
  }

  using partial_to_typed_result_type = typename hostrpc::either<
      typename traits::partial_to_typed_trait<Friend, SelfType, false>::type,
      typename traits::partial_to_typed_trait<Friend, SelfType, true>::type>;

  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator partial_to_typed_result_type()
  {
    using true_typed_port_t =
        typename traits::partial_to_typed_trait<Friend, SelfType, true>::type;
    using false_typed_port_t =
        typename traits::partial_to_typed_trait<Friend, SelfType, false>::type;

    const UnderlyingType p = this->disassemble({});
    const uint32_t v = p.get<0>();
    const bool s = p.get<1>();

    if (s)
      {
        return either<true_typed_port_t, false_typed_port_t>::Left(
            true_typed_port_t::reconstitute({}, v));
      }
    else
      {
        return either<true_typed_port_t, false_typed_port_t>::Right(
            false_typed_port_t::reconstitute({}, v));
      }
  }

  // Call into the above, via either knowing how to swap its branches
  HOSTRPC_ANNOTATE
  HOSTRPC_CALL_ON_LIVE
  HOSTRPC_SET_TYPESTATE(consumed)
  operator typename hostrpc::either<
      typename traits::partial_to_typed_trait<Friend, SelfType, true>::type,
      typename traits::partial_to_typed_trait<Friend, SelfType, false>::type>()
  {
    typename hostrpc::either<
        typename traits::partial_to_typed_trait<Friend, SelfType, false>::type,
        typename traits::partial_to_typed_trait<Friend, SelfType, true>::type>
        either = *this;
    return either;
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

template <typename Friend, unsigned I, unsigned O>
HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE HOSTRPC_SET_TYPESTATE(consumed)
    typed_port_impl_t<Friend, I, O>::operator typename traits::
        typed_to_partial_trait<Friend, SelfType>::type()
{
  using PartialTrait = traits::typed_to_partial_trait<Friend, SelfType>;
  UnderlyingType v = this->disassemble({});
  typename PartialTrait::type::UnderlyingType partial = {v,
                                                         PartialTrait::state()};
  return PartialTrait::type::reconstitute({}, partial);
}

// Can convert losslessly between partial and typed ports. These are mostly
// implemented as user defined conversions to interact reasonably with
// typestate, but that requires spelling out the types relatively frequently.

template <typename Friend, unsigned S>
HOSTRPC_ANNOTATE
    typename partial_port_impl_t<Friend, S>::partial_to_typed_result_type
    partial_to_typed(partial_port_impl_t<Friend, S> &&port)
{
  return port;
}

template <typename Friend, unsigned I, unsigned O>
HOSTRPC_ANNOTATE typename traits::typed_to_partial_trait<
    Friend, typed_port_impl_t<Friend, I, O>>::type
typed_to_partial(typed_port_impl_t<Friend, I, O> &&port)
{
  return port;
}

template <typename Friend, unsigned IA, unsigned OA, unsigned IB, unsigned OB>
HOSTRPC_ANNOTATE either<typename traits::typed_to_partial_trait<
                            Friend, typed_port_impl_t<Friend, IA, OA>>::type,
                        typename traits::typed_to_partial_trait<
                            Friend, typed_port_impl_t<Friend, IB, OB>>::type>
typed_to_partial(either<typed_port_impl_t<Friend, IA, OA>,
                        typed_port_impl_t<Friend, IB, OB>> &&port)
{
  using true_typed_port_t = typename traits::typed_to_partial_trait<
      Friend, typed_port_impl_t<Friend, IA, OA>>::type;
  using false_typed_port_t = typename traits::typed_to_partial_trait<
      Friend, typed_port_impl_t<Friend, IB, OB>>::type;
  using result_type = either<true_typed_port_t, false_typed_port_t>;
  return port.template visit<result_type>(
      [](typed_port_impl_t<Friend, IA, OA> &&port) -> result_type {
        return result_type::Left(typed_to_partial(cxx::move(port)));
      },
      [](typed_port_impl_t<Friend, IB, OB> &&port) -> result_type {
        return result_type::Right(typed_to_partial(cxx::move(port)));
      });
}

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
