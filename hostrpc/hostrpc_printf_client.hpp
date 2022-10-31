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
  hostrpc::port_t port_escape(hostrpc::typed_port_impl_t<F, I, O> && port)
  {
    uint32_t v = port;
    port.disown();
    return static_cast<hostrpc::port_t>(v);
 }

}  // namespace

// Functions implemented out of header. printf resolves to multiple calls to
// these. Some implemented on gcn. All should probably be implemented on
// gcn/ptx/x64
__PRINTF_API_EXTERNAL hostrpc::port_t  __printf_print_start(const char *fmt);
__PRINTF_API_EXTERNAL int __printf_print_end(hostrpc::port_t port);

// simple types
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int32(hostrpc::port_t port,
                                                       int32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint32(hostrpc::port_t port,
                                                        uint32_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int64(hostrpc::port_t port,
                                                       int64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint64(hostrpc::port_t port,
                                                        uint64_t x);
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_double(hostrpc::port_t port,
                                                        double x);

// print the address of the argument on the gpu
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_void(hostrpc::port_t port,
                                                      const void *x);

// copy null terminated string starting at x, print the string
template <typename F, unsigned I, unsigned O>
__PRINTF_API_EXTERNAL  hostrpc::port_t __printf_pass_element_cstr(hostrpc::typed_port_impl_t<F,I,O> && port,
                                                      const char *x);

// implement %n specifier, may need one per sizeof target
__PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_write_int64(hostrpc::port_t port,
                                                             int64_t *x);
template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_print_start(T *client, const char *fmt)
{
  auto active_threads = platform::active_threads(); // warning, not valid on volta
  // TODO: Check a port is eventually available
  // Otherwise need to report failure via the return value from printf

  typename
    T::template typed_port_t<0, 0> tport = client->template rpc_open_typed_port<0,0>(active_threads);


  __printf_print_start_t inst;
  send_by_copy<__printf_print_start_t> f(&inst);

  typename T::template typed_port_t<0, 1> tport2 =
    client->rpc_port_send(active_threads, hostrpc::cxx::move(tport), f);
  

  hostrpc::port_t port =
    __printf_pass_element_cstr(hostrpc::cxx::move(tport2), fmt);

  return port;
  //return __printf_pass_element_cstr(port, fmt);

}

template <typename T>
__PRINTF_API_INTERNAL int __printf_print_end(T *client, hostrpc::port_t port)
{
  assert(port != hostrpc::port_t::unavailable);
  auto active_threads = platform::active_threads();
  {
    __printf_print_end_t inst;
    send_by_copy<__printf_print_end_t> f(&inst);
    client->rpc_port_send(active_threads, port, f);
  }

  client->rpc_port_wait_for_result(active_threads, port);

  client->rpc_port_discard_result(active_threads, port);

  client->rpc_close_port(active_threads, port);

  return 0;  // should be return code from printf
}

template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_pass_element_uint64(T *client,
                                                        hostrpc::port_t port,
                                                        uint64_t v)
{
  assert(port != hostrpc::port_t::unavailable);
  auto active_threads = platform::active_threads();
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(active_threads, port, f);
  return port;
}

template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_pass_element_double(T *client,
                                                        hostrpc::port_t port,
                                                        double v)
{
  assert(port != hostrpc::port_t::unavailable);
  auto active_threads = platform::active_threads();
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_double, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(active_threads, port, f);
  return port;
}

template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_pass_element_void(T *client, hostrpc::port_t port,
                                                      const void *v)
{
  assert(port != hostrpc::port_t::unavailable);
  _Static_assert(sizeof(const void *) == 8, "");
  auto active_threads = platform::active_threads();
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, c);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(active_threads, port, f);
  return port;
}

template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_pass_element_cstr(T *client,
                                                                 typename T:: typed_port_t && tport,
                                                                 const char *str)
{
  hostrpc::port_t port = port_escape(hostrpc::cxx::move(tport));
  assert(port != hostrpc::port_t::unavailable);
  auto active_threads = platform::active_threads();
  uint64_t L = __printf_strlen(str);

  const constexpr size_t w = __printf_pass_element_cstr_t::width;

  // this appears to behave poorly when different threads make different numbers
  // of calls
  uint64_t chunks = L / w;
  uint64_t remainder = L - (chunks * w);
  for (uint64_t c = 0; c < chunks; c++)
    {
      __printf_pass_element_cstr_t inst(&str[c * w], w);
      send_by_copy<__printf_pass_element_cstr_t> f(&inst);
      client->rpc_port_send(active_threads, port, f);
    }

  // remainder < width, possibly zero. sending even when zero ensures null
  // terminated.
  {
    __printf_pass_element_cstr_t inst(&str[chunks * w], remainder);
    send_by_copy<__printf_pass_element_cstr_t> f(&inst);
    client->rpc_port_send(active_threads, port, f);
  }

  return port;
}

template <typename T>
__PRINTF_API_INTERNAL hostrpc::port_t __printf_pass_element_write_int64(T *client,
                                                             hostrpc::port_t port,
                                                             int64_t *x)
{
  assert(port != hostrpc::port_t::unavailable);
  auto active_threads = platform::active_threads();
  __printf_pass_element_write_t inst;
  inst.payload = 42;
  send_by_copy<__printf_pass_element_write_t> f(&inst);
  client->rpc_port_send(active_threads, port, f);

  // need to recv to get the result
  recv_by_copy<__printf_pass_element_write_t> r(&inst);
  client->rpc_port_recv(active_threads, port, r);
  *x = inst.payload;
  return port;
}

#define HOSTRPC_PRINTF_INSTANTIATE_CLIENT(TYPE, EXPR)                    \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_print_start(const char *fmt) \
  {                                                                           \
    return __printf_print_start(EXPR, fmt);                                   \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL int __printf_print_end(hostrpc::port_t port)                 \
  {                                                                           \
    return __printf_print_end(EXPR, port);                                    \
  }                                                                           \
                                                                              \
  /* These may want to be their own functions, for now delegate to u64 */     \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int32(hostrpc::port_t port,       \
                                                         int32_t v)           \
  {                                                                           \
    int64_t w = v;                                                            \
    return __printf_pass_element_int64(port, w);                              \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint32(hostrpc::port_t port,      \
                                                          uint32_t v)         \
  {                                                                           \
    uint64_t w = v;                                                           \
    return __printf_pass_element_uint64(port, w);                             \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_int64(hostrpc::port_t port,       \
                                                         int64_t v)           \
  {                                                                           \
    uint64_t c;                                                               \
    __builtin_memcpy(&c, &v, 8);                                              \
    return __printf_pass_element_uint64(port, c);                             \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_uint64(hostrpc::port_t port,      \
                                                          uint64_t v)         \
  {                                                                           \
    return __printf_pass_element_uint64(EXPR, port, v);                       \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_double(hostrpc::port_t port,      \
                                                          double v)           \
  {                                                                           \
    return __printf_pass_element_double(EXPR, port, v);                       \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_void(hostrpc::port_t port,        \
                                                        const void *v)        \
  {                                                                           \
    return __printf_pass_element_void(EXPR, port, v);                                \
  }                                                                           \
                                     \
 template <unsigned I, unsigned O>                         \
  __PRINTF_API_EXTERNAL   hostrpc::port_t __printf_pass_element_cstr(TYPE::typed_port_t<I, O> && port, \
                                                        const char *str)      \
  {                                                                           \
    return __printf_pass_element_cstr(EXPR, port, str);                              \
  }                                                                           \
                                                                              \
  __PRINTF_API_EXTERNAL hostrpc::port_t __printf_pass_element_write_int64(hostrpc::port_t port, \
                                                               int64_t *x)    \
  {                                                                           \
    return __printf_pass_element_write_int64(EXPR, port, x);                         \
  }

#endif
