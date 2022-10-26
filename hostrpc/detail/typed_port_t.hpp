#ifndef TYPED_PORT_T_HPP_INCLUDED
#define TYPED_PORT_T_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "cxx.hpp"

namespace hostrpc
{
// This is essentially a uint32_t used to index into the arrays within the state
// machine. It tracks the inbox and outbox state in the template parameters
// and detects use-after-move and missing calls to close_port using the
// clang typestate machinery.

// There's going to be a class to represent an instance of the state machine
template <typename Friend, unsigned I, unsigned O>
class typed_port_impl_t;

#define HOSTRPC_USE_TYPESTATE 1

#if defined(__OPENCL_C_VERSION__)
// May be able to make this work, need work out why opencl is upset about
// the definition of cxx::move. Looks like tagging __private or __generic
// on some of the types is done inconsistently
#undef HOSTRPC_USE_TYPESTATE
#define HOSTRPC_USE_TYPESTATE 0
#endif

#if HOSTRPC_USE_TYPESTATE
#define HOSTRPC_RETURN_CONSUMED __attribute__((return_typestate(consumed)))
#define HOSTRPC_RETURN_UNKNOWN __attribute__((return_typestate(unknown)))
  
#define HOSTRPC_CONSUMED_ARG \
  __attribute__((param_typestate(unconsumed))) HOSTRPC_RETURN_CONSUMED
#define HOSTRPC_CREATED_RES __attribute__((return_typestate(unconsumed)))
#define HOSTRPC_CALL_ON_LIVE __attribute__((callable_when(unconsumed)))
#define HOSTRPC_CALL_ON_DEAD __attribute__((callable_when(consumed)))
#define HOSTRPC_CONSUMABLE_CLASS __attribute__((consumable(unconsumed)))
#define HOSTRPC_SET_TYPESTATE(X) __attribute__((set_typestate(X)))
#else
#define HOSTRPC_RETURN_CONSUMED
#define HOSTRPC_CONSUMED_ARG
#define HOSTRPC_CREATED_RES
#define HOSTRPC_CALL_ON_LIVE
#define HOSTRPC_CALL_ON_DEAD
#define HOSTRPC_CONSUMABLE_CLASS
#define HOSTRPC_SET_TYPESTATE(X)
#endif

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
}  // namespace cxx
#endif

template <typename Friend, unsigned I, unsigned O>
class HOSTRPC_CONSUMABLE_CLASS typed_port_impl_t
{
 private:
  friend Friend;  // the state machine

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
  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE constexpr operator uint32_t() const
  {
    return value;
  }

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

  HOSTRPC_ANNOTATE HOSTRPC_CREATED_RES constexpr typed_port_impl_t(uint32_t v)
      : value(v)
  {
    static_assert((I <= 1) && (O <= 1), "");
  }

  HOSTRPC_ANNOTATE HOSTRPC_CALL_ON_LIVE void drop() { kill(); }
  typed_port_impl_t(const typed_port_impl_t &) = delete;
  typed_port_impl_t &operator=(const typed_port_impl_t &) = delete;
  uint32_t value;
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
}  // namespace cxx
#endif
}  // namespace hostrpc

#endif
