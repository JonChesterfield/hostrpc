#ifndef HOSTRPC_PRINTF_SERVER_HPP_INCLUDED
#define HOSTRPC_PRINTF_SERVER_HPP_INCLUDED

#include "hostrpc_printf.h"
#include "platform.hpp"

#include "hostrpc_printf_api_macro.h"
#include "hostrpc_printf_common.hpp"

#if (HOSTRPC_HOST)
// Server is only implemented for the host at present

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

namespace
{
HOSTRPC_ANNOTATE bool fs_contains_nul(const char *str, size_t limit)
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

// Return 'true' if it did work, 'false' if line->element[0] did not match
HOSTRPC_ANNOTATE inline bool operate_printf_handle(unsigned c,
                                                   hostrpc::cacheline_t *line,
                                                   print_wip &thread_print,
                                                   bool verbose)
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

  void operator()(hostrpc::port_t port, hostrpc::page_t *page)
  {
    uint32_t slot = static_cast<uint32_t>(port);
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

  void operator()(hostrpc::port_t, hostrpc::page_t *page)
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

#endif
