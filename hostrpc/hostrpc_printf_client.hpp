#ifndef HOSTRPC_PRINTF_CLIENT_HPP_INCLUDED
#define HOSTRPC_PRINTF_CLIENT_HPP_INCLUDED

#include "hostrpc_printf_api_macro.h"
#include "hostrpc_printf_common.hpp"
#include "platform.hpp"

namespace
{
// TODO: When native_width < 64, this wastes a lot of space
// The page abstraction being indexed by thread is messy,
template <typename T>
struct send_by_copy
{
  static_assert(sizeof(T) == 64, "");
  HOSTRPC_ANNOTATE send_by_copy(T *i) : i(i) {}
  T *i;

  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    unsigned id = platform::get_lane_id();
    hostrpc::cacheline_t *dline = &page->cacheline[id];
    __builtin_memcpy(dline, i, 64);
  }
};

template <typename T>
struct recv_by_copy
{
  static_assert(sizeof(T) == 64, "");
  HOSTRPC_ANNOTATE recv_by_copy(T *i) : i(i) {}
  T *i;

  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, hostrpc::page_t *page)
  {
    unsigned id = platform::get_lane_id();
    hostrpc::cacheline_t *dline = &page->cacheline[id];
    __builtin_memcpy(i, dline, 64);
  }
};

template <typename F, unsigned I, unsigned O>
hostrpc::port_t port_escape(hostrpc::typed_port_impl_t<F, I, O> &&port)
{
  uint32_t v = port;
  port.disown();
  return static_cast<hostrpc::port_t>(v);
}

}  // namespace

// copy null terminated string starting at x, print the string
template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_cstr(T *client,
                           typename T::template typed_port_t<0, 0> &&tport,
                           const char *str);

// simple types
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int32(
    hostrpc::port_t port, int32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint32(
    hostrpc::port_t port, uint32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int64(
    hostrpc::port_t port, int64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint64(
    hostrpc::port_t port, uint64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_double(
    hostrpc::port_t port, double x);

// print the address of the argument on the gpu
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_void(
    hostrpc::port_t port, const void *x);

// copy null terminated string starting at x, print the string

// implement %n specifier, may need one per sizeof target
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_write_int64(
    hostrpc::port_t port, int64_t *x);
template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_print_start(T *client, const char *fmt)
{
  auto active_threads =
      platform::active_threads();  // warning, not valid on volta
  // TODO: Check a port is eventually available
  // Otherwise need to report failure via the return value from printf

  typename T::template typed_port_t<0, 0> tport =
      client->template rpc_open_typed_port<0, 0>(active_threads);

  __printf_print_start_t inst;
  send_by_copy<__printf_print_start_t> f(&inst);

  typename T::template typed_port_t<0, 1> tport2 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(tport), f);

  typename T::template typed_port_t<1, 1> tport3 =
      client->rpc_port_wait(active_threads, hostrpc::cxx::move(tport2));

  typename T::template typed_port_t<1, 0> tport4 =
      client->rpc_port_discard_result(active_threads,
                                      hostrpc::cxx::move(tport3));

  typename T::template typed_port_t<0, 0> tport5 =
      client->rpc_port_wait(active_threads, hostrpc::cxx::move(tport4));

  typename T::template typed_port_t<0, 0> tport6 =
      __printf_pass_element_cstr<T>(client, hostrpc::cxx::move(tport5), fmt);

  return hostrpc::cxx::move(tport6);
}

template <typename T>
__PRINTF_API_INTERNAL int __printf_print_end(
    T *client, typename T::template typed_port_t<0, 0> &&port0)
{
  auto active_threads = platform::active_threads();

  __printf_print_end_t inst;
  send_by_copy<__printf_print_end_t> f(&inst);
  auto port1 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(port0), f);

  auto port2 = client->rpc_port_wait_for_result(active_threads,
                                                hostrpc::cxx::move(port1));

  auto port3 = client->rpc_port_discard_result(active_threads,
                                               hostrpc::cxx::move(port2));

  client->rpc_close_port(active_threads, hostrpc::cxx::move(port3));

  return 0;  // should be return code from printf
}

template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_uint64(T *client,
                             typename T::template typed_port_t<0, 0> &&port0,
                             uint64_t v)
{
  auto active_threads = platform::active_threads();
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  typename T::template typed_port_t<0, 1> port1 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(port0), f);

  typename T::template typed_port_t<0, 0> port2 =
      client->rpc_port_wait_until_available(active_threads,
                                            hostrpc::cxx::move(port1));

  return hostrpc::cxx::move(port2);
}

template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_double(T *client,
                             typename T::template typed_port_t<0, 0> &&port0,
                             double v)
{
  auto active_threads = platform::active_threads();
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_double, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  typename T::template typed_port_t<0, 1> port1 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(port0), f);

  typename T::template typed_port_t<0, 0> port2 =
      client->rpc_port_wait_until_available(active_threads,
                                            hostrpc::cxx::move(port1));

  return hostrpc::cxx::move(port2);
}

template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_void(T *client,
                           typename T::template typed_port_t<0, 0> &&port0,
                           const void *v)
{
  _Static_assert(sizeof(const void *) == 8, "");
  auto active_threads = platform::active_threads();
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, c);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);

  typename T::template typed_port_t<0, 1> port1 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(port0), f);

  typename T::template typed_port_t<0, 0> port2 =
      client->rpc_port_wait_until_available(active_threads,
                                            hostrpc::cxx::move(port1));

  return hostrpc::cxx::move(port2);
}

template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_cstr(T *client,
                           typename T::template typed_port_t<0, 0> &&port,
                           const char *str)
{
  auto active_threads = platform::active_threads();
  uint64_t L = __printf_strlen(str);

  const constexpr size_t w = __printf_pass_element_cstr_t::width;

  typename T::template typed_port_t<0, 0> ready =
      client->rpc_port_wait_until_available(active_threads,
                                            hostrpc::cxx::move(port));
  ready.unconsumed();

  // this appears to behave poorly when different threads make different numbers
  // of calls
  uint64_t chunks = L / w;
  uint64_t remainder = L - (chunks * w);

  ready.unconsumed();

  // Goto typechecked and for loop didn't.
  // Rotated loop - if + do/while - also didn't typecheck
  // for (uint64_t c = 0; c < chunks; c++)
  {
    uint64_t c = 0;
  loop:
    if (c == chunks) goto done;
    {
      ready.unconsumed();
      __printf_pass_element_cstr_t inst(&str[c * w], w);
      send_by_copy<__printf_pass_element_cstr_t> f(&inst);
      ready.unconsumed();
      typename T::template typed_port_t<0, 1> tmp =
          client->rpc_port_send(active_threads, hostrpc::cxx::move(ready), f);
      tmp.unconsumed();
      ready.consumed();
      ready = client->rpc_port_wait_until_available(active_threads,
                                                    hostrpc::cxx::move(tmp));
      ready.unconsumed();
    }

    c++;
    goto loop;
  done:;
  }

  // remainder < width, possibly zero. sending even when zero ensures null
  // terminated.
  {
    __printf_pass_element_cstr_t inst(&str[chunks * w], remainder);
    send_by_copy<__printf_pass_element_cstr_t> f(&inst);
    auto tmp =
        client->rpc_port_send(active_threads, hostrpc::cxx::move(ready), f);

    return client->rpc_port_wait_until_available(active_threads,
                                                 hostrpc::cxx::move(tmp));
  }
}

template <typename T>
__PRINTF_API_INTERNAL typename T::template typed_port_t<0, 0>
__printf_pass_element_write_int64(
    T *client, typename T::template typed_port_t<0, 0> &&port0, int64_t *x)
{
  auto active_threads = platform::active_threads();

  __printf_pass_element_write_t inst;
  inst.payload = 42;

  send_by_copy<__printf_pass_element_write_t> f(&inst);
  typename T::template typed_port_t<0, 1> port1 =
      client->rpc_port_send(active_threads, hostrpc::cxx::move(port0), f);

  // need to recv to get the result
  recv_by_copy<__printf_pass_element_write_t> r(&inst);
  typename T::template typed_port_t<1, 0> port2 =
      client->rpc_port_recv(active_threads, hostrpc::cxx::move(port1), r);
  *x = inst.payload;

  typename T::template typed_port_t<0, 0> port3 =
      client->rpc_port_wait_until_available(active_threads,
                                            hostrpc::cxx::move(port2));

  return hostrpc::cxx::move(port3);
}

#define HOSTRPC_PRINTF_INSTANTIATE_CLIENT(TYPE, EXPR)                          \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_print_start(const char *fmt)  \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> res =                           \
        __printf_print_start(EXPR, fmt);                                       \
    return port_escape(hostrpc::cxx::move(res));                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL int __printf_print_end(hostrpc::port_t port)           \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    return __printf_print_end(EXPR, hostrpc::cxx::move(p));                    \
  }                                                                            \
                                                                               \
  /* These may want to be their own functions, for now delegate to u64 */      \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int32(           \
      hostrpc::port_t port, int32_t v)                                         \
  {                                                                            \
    int64_t w = v;                                                             \
    return __printf_pass_element_int64(port, w);                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint32(          \
      hostrpc::port_t port, uint32_t v)                                        \
  {                                                                            \
    uint64_t w = v;                                                            \
    return __printf_pass_element_uint64(port, w);                              \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int64(           \
      hostrpc::port_t port, int64_t v)                                         \
  {                                                                            \
    uint64_t c;                                                                \
    __builtin_memcpy(&c, &v, 8);                                               \
    return __printf_pass_element_uint64(port, c);                              \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint64(          \
      hostrpc::port_t port, uint64_t v)                                        \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    auto res = __printf_pass_element_uint64(EXPR, hostrpc::cxx::move(p), v);   \
    return port_escape(hostrpc::cxx::move(res));                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_double(          \
      hostrpc::port_t port, double v)                                          \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    auto res = __printf_pass_element_double(EXPR, hostrpc::cxx::move(p), v);   \
    return port_escape(hostrpc::cxx::move(res));                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_void(            \
      hostrpc::port_t port, const void *v)                                     \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    auto res = __printf_pass_element_void(EXPR, hostrpc::cxx::move(p), v);     \
    return port_escape(hostrpc::cxx::move(res));                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_cstr(            \
      hostrpc::port_t port, const char *str)                                   \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    auto res = __printf_pass_element_cstr(EXPR, hostrpc::cxx::move(p), str);   \
    return port_escape(hostrpc::cxx::move(res));                               \
  }                                                                            \
                                                                               \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_write_int64(     \
      hostrpc::port_t port, int64_t *x)                                        \
  {                                                                            \
    typename TYPE::template typed_port_t<0, 0> p(static_cast<uint32_t>(port)); \
    auto res =                                                                 \
        __printf_pass_element_write_int64(EXPR, hostrpc::cxx::move(p), x);     \
    return port_escape(hostrpc::cxx::move(res));                               \
  }

#endif
