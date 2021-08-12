#ifndef HOSTRPC_PRINTF_SERVER_HPP_INCLUDED
#define HOSTRPC_PRINTF_SERVER_HPP_INCLUDED

#include "hostrpc_printf.h"
#include "hostrpc_printf_enable.h"

#if (HOSTRPC_AMDGCN)

namespace
{
template <typename T>
struct send_by_copy
{
  static_assert(sizeof(T) == 64, "");
  HOSTRPC_ANNOTATE send_by_copy(T *i) : i(i) {}
  T *i;

  HOSTRPC_ANNOTATE void operator()(hostrpc::page_t *page)
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

  HOSTRPC_ANNOTATE void operator()(hostrpc::page_t *page)
  {
    unsigned id = platform::get_lane_id();
    hostrpc::cacheline_t *dline = &page->cacheline[id];
    __builtin_memcpy(i, dline, 64);
  }
};
}  // namespace


template <typename T>
__PRINTF_API_INTERNAL uint32_t __printf_print_start(T * client, const char *fmt)
{
    uint32_t port = client->rpc_open_port();
  if (port == UINT32_MAX)
    {
      // failure
      UINT32_MAX;
    }

  {
    __printf_print_start_t inst;
    send_by_copy<__printf_print_start_t> f(&inst);
    client->rpc_port_send(port, f);
  }

  __printf_pass_element_cstr(port, fmt);

  return port;
}

template <typename T>
__PRINTF_API_INTERNAL int __printf_print_end(T * client, uint32_t port)
{
  {
    __printf_print_end_t inst;
    send_by_copy<__printf_print_end_t> f(&inst);
    client->rpc_port_send(port, f);
  }

  client->rpc_port_wait_for_result(port);

  client->rpc_close_port(port);
  return 0;  // should be return code from printf
}



template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_uint64(T* client, uint32_t port,
                                                        uint64_t v)
{
  __printf_pass_element_scalar_t inst(func___printf_pass_element_uint64, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(port, f);
}

template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_double(T* client,uint32_t port, double v)
{
  __printf_pass_element_scalar_t inst(func___printf_pass_element_double, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(port, f);
}


template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_void(T* client, uint32_t port,
                                                      const void *v)
{
  _Static_assert(sizeof(const void *) == 8, "");
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  __printf_pass_element_scalar_t inst(func___printf_pass_element_uint64, c);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(port, f);
}


template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_cstr(T* client, uint32_t port,
                                                      const char *str)
{
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
      client->rpc_port_send(port, f);
    }

  // remainder < width, possibly zero. sending even when zero ensures null
  // terminated.
  {
    __printf_pass_element_cstr_t inst(&str[chunks * w], remainder);
    send_by_copy<__printf_pass_element_cstr_t> f(&inst);
    client->rpc_port_send(port, f);
  }
}

template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_write_int64(T * client, uint32_t port,
                                                             int64_t *x)
{
  __printf_pass_element_write_t inst;
  inst.payload = 42;
  send_by_copy<__printf_pass_element_write_t> f(&inst);
  client->rpc_port_send(port, f);

  // need to recv to get the result
  recv_by_copy<__printf_pass_element_write_t> r(&inst);
  client->rpc_port_recv(port, r);
  *x = inst.payload;
}

#endif
#endif
