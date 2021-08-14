#ifndef HOSTRPC_PRINTF_SERVER_HPP_INCLUDED
#define HOSTRPC_PRINTF_SERVER_HPP_INCLUDED

#include "detail/platform.hpp"
#include "hostrpc_printf.h"
#include "hostrpc_printf_enable.h"

namespace
{
HOSTRPC_ANNOTATE __attribute__((unused)) size_t fs_strnlen(const char *str,
                                                           size_t limit)
{
  for (size_t i = 0; i < limit; i++)
    {
      if (str[i] == '\0')
        {
          return i;
        }
    }
  return limit;
}

HOSTRPC_ANNOTATE __attribute__((unused)) bool fs_contains_nul(const char *str,
                                                              size_t limit)
{
  bool r = false;
  for (size_t i = 0; i < limit; i++)
    {
      if (str[i] == '\0')
        {
          r = true;
        }
    }
  return r;
}

}  // namespace

#if (HOSTRPC_HOST)
#include "incprintf.hpp"

#include <array>
#include <vector>

namespace
{
// server cases are much easier to write if one port is used
// for multiple packets
struct print_wip
{
  incr formatter;
  int bytes_written = 0;  // negative on error

  void update_bytes_written(int x)
  {
    if (bytes_written < 0)
      {
        return;
      }
    if (x < 0)
      {
        bytes_written = x;
        return;
      }
    bytes_written += x;
  }
  // first expected to be cstr for format
  struct field
  {
    field() = default;

    uint64_t tag = hostrpc_printf_print_nop;

    static field cstr()
    {
      field r;
      r.tag = hostrpc_printf_pass_element_cstr;
      return r;
    }
  };

  void clear() { acc = {}; }

  field acc;
};

using print_buffer_t = std::vector<std::array<print_wip, 64> >;

// Return 'true' if it did work, 'false' if line->element[0] did not match
bool operate_printf_handle(unsigned c, hostrpc::cacheline_t *line,
                           print_wip &thread_print, bool verbose)
{
  (void)verbose;
  uint64_t ID = line->element[0];
  const bool prefix_thread_id = false;
  switch (ID)
    {
      case hostrpc_printf_print_nop:
        {
          return true;
        }

      case hostrpc_printf_print_start:
        {
          // fprintf(stderr, ".");
          thread_print.formatter = incr{};
          if (prefix_thread_id)
            thread_print.formatter.append_cstr_section<7>("[%.2u] ");
          thread_print.clear();
          thread_print.acc = print_wip::field::cstr();
          thread_print.acc.tag = hostrpc_printf_pass_element_cstr;
          return true;
        }

      case hostrpc_printf_print_end:
        {
          std::vector<char> r = thread_print.formatter.finalize();
          printf("%s", r.data());
          thread_print.acc.tag = hostrpc_printf_print_nop;
          return true;
        }

      case hostrpc_printf_pass_element_cstr:
        {
          __printf_pass_element_cstr_t *p =
              reinterpret_cast<__printf_pass_element_cstr_t *>(
                  &line->element[0]);

          if (thread_print.acc.tag == hostrpc_printf_print_nop)
            {
              // starting new cstr
              thread_print.acc = print_wip::field::cstr();
              thread_print.formatter.accumulator.clear();
            }

          if (thread_print.acc.tag == hostrpc_printf_pass_element_cstr)
            {
              thread_print.formatter
                  .append_cstr_section<__printf_pass_element_cstr_t::width>(
                      p->payload);

              if (fs_contains_nul(p->payload,
                                  __printf_pass_element_cstr_t::width))
                {
                  thread_print.formatter.accumulator.push_back(
                      '\0');  // assumed by formatter

                  const char *s = thread_print.formatter.accumulator.data();
                  if (thread_print.formatter.have_format())
                    {
                      thread_print.update_bytes_written(
                          thread_print.formatter
                              .__printf_pass_element_T<const char *>(s));
                    }
                  else
                    {
                      thread_print.formatter.set_format(s);
                      if (prefix_thread_id)
                        {
                          thread_print.formatter
                              .__printf_pass_element_T<unsigned>(c);
                        }
                    }

                  // end of string

                  assert(thread_print.acc.tag != hostrpc_printf_print_nop);
                  thread_print.acc = {};
                }
            }
          else
            {
              printf("invalid print cstr\n");
            }

          return true;
        }

      case hostrpc_printf_pass_element_scalar:
        {
          __printf_pass_element_scalar_t *p =
              reinterpret_cast<__printf_pass_element_scalar_t *>(
                  &line->element[0]);

          switch (p->Type)
            {
              case hostrpc_printf_pass_element_uint64:
                {
                  if (thread_print.acc.tag == hostrpc_printf_print_nop)
                    {
                      thread_print.update_bytes_written(
                          thread_print.formatter
                              .__printf_pass_element_T<uint64_t>(p->payload));

                      thread_print.acc.tag = hostrpc_printf_pass_element_uint64;
                      thread_print.acc = {};
                    }
                  else
                    {
                      printf("invalid pass u64\n");  // missing trailing null
                                                     // on the cstr?
                    }
                  break;
                }
              case hostrpc_printf_pass_element_double:
                {
                  if (thread_print.acc.tag == hostrpc_printf_print_nop)
                    {
                      thread_print.update_bytes_written(
                          thread_print.formatter
                              .__printf_pass_element_T<double>(p->payload));

                      thread_print.acc.tag = hostrpc_printf_pass_element_double;
                      thread_print.acc = {};
                    }
                  else
                    {
                      printf("invalid pass double\n");
                    }
                  break;
                }
              default:
                {
                  printf("unimplemented scalar element: %lu\n", p->Type);
                  break;
                }
            }

          return true;
        }

      case hostrpc_printf_pass_element_write_int64:
        {
          __printf_pass_element_write_t *p =
              reinterpret_cast<__printf_pass_element_write_t *>(
                  &line->element[0]);

          if (thread_print.acc.tag == hostrpc_printf_print_nop)
            {
              if (0)
                printf("Got passed a write_int64, initial value %lu (acc %d)\n",
                       p->payload, thread_print.bytes_written);
              thread_print.update_bytes_written(
                  thread_print.formatter.__printf_pass_element_T<int64_t *>(
                      &p->payload));
              p->payload = thread_print.bytes_written;
              if (0)
                printf("Got passed a write_int64, later value %lu (acc %d)\n",
                       p->payload, thread_print.bytes_written);
              thread_print.acc.tag = hostrpc_printf_pass_element_write_int64;
              thread_print.acc = {};
            }
          else
            {
              printf("invalid pass write i64\n");
            }
          return true;
        }
    }

  return false;
}

struct operate
{
  print_buffer_t *print_buffer = nullptr;
  operate(print_buffer_t *print_buffer) : print_buffer(print_buffer) {}
  operate() = default;

  void perthread(unsigned c, hostrpc::cacheline_t *line,
                 print_wip &thread_print, bool verbose)
  {
    if (operate_printf_handle(c, line, thread_print, verbose))
      {
        return;
      }
    else
      {
        printf("Unhandled op: %lu\n", line->element[0]);
        return;
      }
  }

  void operator()(uint32_t slot, hostrpc::page_t *page)
  {
    const bool verbose = false;
    if (verbose) fprintf(stderr, "Invoked operate on slot %u\n", slot);

    std::array<print_wip, 64> &print_slot_buffer = (*print_buffer)[slot];

    for (unsigned c = 0; c < 64; c++)
      {
        perthread(c, &page->cacheline[c], print_slot_buffer[c], verbose);
      }
  }
};

struct clear
{
  clear() = default;

  void operator()(uint32_t, hostrpc::page_t *page)
  {
    for (uint64_t c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t *line = &page->cacheline[c];
        for (uint64_t e = 0; e < 8; e++)
          {
            line->element[e] = 0;
          }
      }
  }
};

}  // namespace

#endif

#if (HOSTRPC_AMDGCN)

namespace
{
template <typename T>
struct send_by_copy
{
  static_assert(sizeof(T) == 64, "");
  HOSTRPC_ANNOTATE send_by_copy(T *i) : i(i) {}
  T *i;

  HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t *page)
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

  HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t *page)
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
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, v);
  send_by_copy<__printf_pass_element_scalar_t> f(&inst);
  client->rpc_port_send(port, f);
}

template <typename T>
__PRINTF_API_INTERNAL void __printf_pass_element_double(T* client,uint32_t port, double v)
{
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_double, v);
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
  __printf_pass_element_scalar_t inst(hostrpc_printf_pass_element_uint64, c);
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
