// This includes platform which contains a stub for printf
#include "x64_gcn_type.hpp"
#undef printf

#include "hostrpc_printf.h"
#include "hostrpc_printf_enable.h"

#include "cxa_atexit.hpp"

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<hostrpc::size_runtime>::client_type hostrpc_x64_gcn_debug_client[1];

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

__PRINTF_API_EXTERNAL uint32_t __printf_print_start(const char *fmt)
{
  uint32_t port = hostrpc_x64_gcn_debug_client[0].rpc_open_port();
  if (port == UINT32_MAX)
    {
      // failure
      UINT32_MAX;
    }

  {
    __printf_print_start_t inst;
    send_by_copy<__printf_print_start_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  __printf_pass_element_cstr(port, fmt);

  return port;
}

__PRINTF_API_EXTERNAL int __printf_print_end(uint32_t port)
{
  {
    __printf_print_end_t inst;
    send_by_copy<__printf_print_end_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  hostrpc_x64_gcn_debug_client[0].rpc_port_wait_for_result(port);

  hostrpc_x64_gcn_debug_client[0].rpc_close_port(port);
  return 0;  // should be return code from printf
}

// These may want to be their own functions, for now delegate to u64

__PRINTF_API_EXTERNAL void __printf_pass_element_int32(uint32_t port, int32_t v)
{
  int64_t w = v;
  return __printf_pass_element_int64(port, w);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_uint32(uint32_t port,
                                                        uint32_t v)
{
  uint64_t w = v;
  return __printf_pass_element_uint64(port, w);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_int64(uint32_t port, int64_t v)
{
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  return __printf_pass_element_uint64(port, c);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_uint64(uint32_t port,
                                                        uint64_t v)
{
  __printf_pass_element_scalar_t inst(func___printf_pass_element_uint64, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_double(uint32_t port, double v)
{
  __printf_pass_element_scalar_t inst(func___printf_pass_element_double, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_void(uint32_t port,
                                                      const void *v)
{
  _Static_assert(sizeof(const void *) == 8, "");
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  __printf_pass_element_scalar_t inst(func___printf_pass_element_uint64, c);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_cstr(uint32_t port,
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
      hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
    }

  // remainder < width, possibly zero. sending even when zero ensures null
  // terminated.
  {
    __printf_pass_element_cstr_t inst(&str[chunks * w], remainder);
    send_by_copy<__printf_pass_element_cstr_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }
}

__PRINTF_API_EXTERNAL void __printf_pass_element_write_int64(uint32_t port,
                                                             int64_t *x)
{
  __printf_pass_element_write_t inst;
  inst.payload = 42;
  send_by_copy<__printf_pass_element_write_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);

  // need to recv to get the result
  recv_by_copy<__printf_pass_element_write_t> r(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_recv(port, r);
  *x = inst.payload;
}

#endif
