#if defined __OPENCL__
// called by test
void example(void);
kernel void __device_example(void) { example(); }
#else

#include "x64_gcn_debug.hpp"
#include "cxa_atexit.hpp"
#include "detail/platform_detect.hpp"
#include "x64_gcn_type.hpp"

namespace
{
size_t fs_strnlen(const char *str, size_t limit)
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

__attribute__((unused)) bool fs_contains_nul(const char *str, size_t limit)
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

size_t fs_strlen(const char *str)
{
  for (size_t i = 0;; i++)
    {
      if (str[i] == '\0')
        {
          return i;
        }
    }
}

enum func_type : uint64_t
{
  func_print_nop = 0,
  func_print_uuu = 1,
  func_print_start = 2,
  func_print_finish = 3,
  func_print_append_str = 4,

  func_piecewise_print_start = 5,
  func_piecewise_print_end = 6,
  func_piecewise_pass_element_cstr = 7,

  func_piecewise_pass_element_scalar = 8,

  func_piecewise_pass_element_int32,
  func_piecewise_pass_element_uint32,
  func_piecewise_pass_element_int64,
  func_piecewise_pass_element_uint64,
  func_piecewise_pass_element_double,
  func_piecewise_pass_element_void,
  func_piecewise_pass_element_write_int32,
  func_piecewise_pass_element_write_int64,

};

struct print_uuu_instance
{
  uint64_t ID = func_print_uuu;
  char fmt[32] = {0};
  uint64_t arg0 = 0;
  uint64_t arg1 = 0;
  uint64_t arg2 = 0;

  print_uuu_instance(const char *fmt_, uint64_t x0, uint64_t x1, uint64_t x2)
      : arg0(x0), arg1(x1), arg2(x2)
  {
    __builtin_memcpy(fmt, fmt_, fs_strnlen(fmt_, sizeof(fmt)));
  }

  print_uuu_instance(const char *d)
  {
    __builtin_memcpy(&ID, d, sizeof(ID));
    d += sizeof(ID);
    __builtin_memcpy(&fmt, d, sizeof(fmt));
    d += sizeof(fmt);
    __builtin_memcpy(&arg0, d, sizeof(arg0));
    d += sizeof(arg0);
    __builtin_memcpy(&arg1, d, sizeof(arg1));
    d += sizeof(arg1);
    __builtin_memcpy(&arg2, d, sizeof(arg2));
    d += sizeof(arg2);
  }
};

struct print_start
{
  uint64_t ID = func_print_start;
  char unused[56];
  print_start() {}
};
struct print_finish
{
  uint64_t ID = func_print_finish;
  uint64_t packets;
  char unused[48];
  print_finish(uint64_t packets) : packets(packets) {}

  print_finish(const char *d)
  {
    __builtin_memcpy(&ID, d, sizeof(ID));
    d += sizeof(ID);
    __builtin_memcpy(&packets, d, sizeof(packets));
    d += sizeof(packets);
  }
};

struct print_append_str
{
  uint64_t ID = func_print_append_str;
  uint64_t start_port;
  uint64_t position;
  char payload[8] = {0};
  char unused[32];
  print_append_str(uint64_t start_port, uint64_t position, const char *str)
      : start_port(start_port), position(position)
  {
    __builtin_memcpy(payload, str, fs_strnlen(str, sizeof(payload)));
  }

  print_append_str(const char *d)
  {
    __builtin_memcpy(&ID, d, sizeof(ID));
    d += sizeof(ID);
    __builtin_memcpy(&start_port, d, sizeof(start_port));
    d += sizeof(start_port);
    __builtin_memcpy(&position, d, sizeof(position));
    d += sizeof(position);
    __builtin_memcpy(&payload, d, sizeof(payload));
    d += sizeof(payload);
  }
};

struct piecewise_print_start_t
{
  uint64_t ID = func_piecewise_print_start;
  char unused[56];
  piecewise_print_start_t() {}
};

struct piecewise_print_end_t
{
  uint64_t ID = func_piecewise_print_end;
  char unused[56];
  piecewise_print_end_t() {}
};

struct piecewise_pass_element_cstr_t
{
  uint64_t ID = func_piecewise_pass_element_cstr;
  enum
  {
    width = 56
  };
  char payload[width];
  piecewise_pass_element_cstr_t(const char *s, size_t N)
  {
    __builtin_memset(payload, 0, width);
    __builtin_memcpy(payload, s, N);
  }
};

struct piecewise_pass_element_scalar_t
{
  uint64_t ID = func_piecewise_pass_element_scalar;
  uint64_t Type;
  uint64_t payload;
  char unused[40];
  piecewise_pass_element_scalar_t(enum func_type type, uint64_t x)
      : Type(type), payload(x)
  {
  }
};

using SZ = hostrpc::size_runtime;

}  // namespace

#if (HOSTRPC_AMDGCN)

// Implementing on gcn, maybe also on x64:
#define API  // __attribute__((noinline))

API uint32_t piecewise_print_start(const char *fmt);
API int piecewise_print_end(uint32_t port);

// simple types
API void piecewise_pass_element_int32(uint32_t port, int32_t x);
API void piecewise_pass_element_uint32(uint32_t port, uint32_t x);
API void piecewise_pass_element_int64(uint32_t port, int64_t x);
API void piecewise_pass_element_uint64(uint32_t port, uint64_t x);
API void piecewise_pass_element_double(uint32_t port, double x);

// copy null terminated string starting at x, print the string
API void piecewise_pass_element_cstr(uint32_t port, const char *x);
// print the address of the argument on the gpu
API void piecewise_pass_element_void(uint32_t port, const void *x);

// implement %n specifier
API void piecewise_pass_element_write_int32(uint32_t port, int32_t *x);
API void piecewise_pass_element_write_int64(uint32_t port, int64_t *x);

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<SZ>::client_type hostrpc_x64_gcn_debug_client[1];

template <typename T>
struct fill_by_copy
{
  static_assert(sizeof(T) == 64, "");
  fill_by_copy(T *i) : i(i) {}
  T *i;

  void operator()(hostrpc::page_t *page)
  {
    unsigned id = platform::get_lane_id();
    hostrpc::cacheline_t *dline = &page->cacheline[id];
    __builtin_memcpy(dline, i, 64);
  }
};

void hostrpc::print_base(const char *str, uint64_t x0, uint64_t x1, uint64_t x2)
{
  const bool via_piecewise = true;

  if (via_piecewise)
    {
      uint32_t port = piecewise_print_start(str);
      if (port == UINT32_MAX)
        {
          return;
        }

      piecewise_pass_element_uint64(port, x0);
      piecewise_pass_element_uint64(port, x1);
      piecewise_pass_element_uint64(port, x2);

      piecewise_print_end(port);
      return;
    }

  struct fill
  {
    fill(print_uuu_instance *i) : i(i) {}
    print_uuu_instance *i;

    void operator()(hostrpc::page_t *page)
    {
      unsigned id = platform::get_lane_id();
      hostrpc::cacheline_t *dline = &page->cacheline[id];
      __builtin_memcpy(dline, i, 64);
    }
  };

  struct use
  {
    void operator()(hostrpc::page_t *){};
  };

  print_uuu_instance i(str, x0, x1, x2);

  fill_by_copy<print_uuu_instance> f(&i);
  use u;
  hostrpc_x64_gcn_debug_client[0].rpc_invoke(f, u);
}

void print_string(const char *str)
{
  uint32_t port = piecewise_print_start(str);
  if (port == UINT32_MAX)
    {
      return;
    }

  piecewise_print_end(port);
}

void piecewise_pass_element_cstr(uint32_t port,
                                 const char *);  // used to pass fmt

uint32_t piecewise_print_start(const char *fmt)
{
  uint32_t port = hostrpc_x64_gcn_debug_client[0].rpc_open_port();
  if (port == UINT32_MAX)
    {
      // failure
      UINT32_MAX;
    }

  {
    piecewise_print_start_t inst;
    fill_by_copy<piecewise_print_start_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  piecewise_pass_element_cstr(port, fmt);

  return port;
}

int piecewise_print_end(uint32_t port)
{
  {
    piecewise_print_end_t inst;
    fill_by_copy<piecewise_print_end_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  hostrpc_x64_gcn_debug_client[0].rpc_close_port(port);
  return 0;  // should be return code from printf
}

void piecewise_pass_element_uint64(uint32_t port, uint64_t v)
{
  piecewise_pass_element_scalar_t inst(func_piecewise_pass_element_uint64, v);
  fill_by_copy<piecewise_pass_element_scalar_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
}

void piecewise_pass_element_cstr(uint32_t port, const char *str)
{
  uint64_t L = fs_strlen(str);

  const constexpr size_t w = piecewise_pass_element_cstr_t::width;

  uint64_t chunks = L / w;
  uint64_t remainder = L - (chunks * w);
  for (uint64_t c = 0; c < chunks; c++)
    {
      piecewise_pass_element_cstr_t inst(&str[c * w], w);
      fill_by_copy<piecewise_pass_element_cstr_t> f(&inst);
      hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
    }

  // remainder < width, possibly zero. sending even when zero ensures null
  // terminated.
  {
    piecewise_pass_element_cstr_t inst(&str[chunks * w], remainder);
    fill_by_copy<piecewise_pass_element_cstr_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }
}

#endif

#if (HOSTRPC_HOST)

#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "incbin.h"
#include "server_thread_state.hpp"

#include <algorithm>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
// server cases are much easier to write if one port is used
// for multiple packets
struct print_wip
{
  // first expected to be cstr for format
  struct field
  {
    field() = default;

    field(int32_t x) : tag(func_piecewise_pass_element_int32)
    {
      int64_t tmp = x;
      __builtin_memcpy(&u64_, &tmp, 8);
    }
    field(int64_t x) : tag(func_piecewise_pass_element_int64)
    {
      __builtin_memcpy(&u64_, &x, 8);
    }
    field(uint32_t x) : tag(func_piecewise_pass_element_uint32) { u64_ = x; }
    field(uint64_t x) : tag(func_piecewise_pass_element_uint64) { u64_ = x; }

    field(double x) : tag(func_piecewise_pass_element_double) { dbl_ = x; }

    static field cstr()
    {
      field r;
      r.tag = func_piecewise_pass_element_cstr;
      return r;
    }

    template <size_t N>
    void append_cstr(const char *s)
    {
      assert(tag == func_piecewise_pass_element_cstr);
      cstr_.insert(cstr_.end(), s, s + N);
    }

    uint64_t tag = func_print_nop;
    uint64_t u64_;
    double dbl_;
    std::vector<char> cstr_;
  };

  uint64_t operator()(size_t i)
  {
    switch (args_[i].tag)
      {
        default:
        case func_print_nop:
          return 0;
        case func_piecewise_pass_element_uint64:
          return args_[i].u64_;
        case func_piecewise_pass_element_cstr:
          return reinterpret_cast<uint64_t>(args_[i].cstr_.data());
      }
  }

  void append_acc()
  {
    assert(acc.tag != func_print_nop);
    args_.push_back(acc);
    acc = {};
  }

  void clear()
  {
    acc = {};
    args_.clear();
  }

  field acc;
  std::vector<field> args_;
};

using buffer_t = std::unordered_map<
    uint32_t, std::array<std::unordered_map<uint64_t, std::string>, 64> >;

using print_buffer_t = std::vector<std::array<print_wip, 64> >;

template <typename ServerType>
struct operate
{
  buffer_t *buffer = nullptr;
  print_buffer_t *print_buffer = nullptr;
  ServerType *ThisServer;
  hostrpc::page_t *start_local_buffer = nullptr;
  operate(buffer_t *buffer, print_buffer_t *print_buffer,
          ServerType *ThisServer)
      : buffer(buffer), print_buffer(print_buffer), ThisServer(ThisServer)
  {
    start_local_buffer = ThisServer->local_buffer;
  }
  operate() = default;

  uint64_t op(hostrpc::page_t *page, unsigned wave)
  {
    hostrpc::cacheline_t *line = &page->cacheline[wave];
    return line->element[0];
  }

  bool uniform(hostrpc::page_t *page)
  {
    bool uniform = true;
    uint64_t first = op(page, 0);
    for (unsigned c = 1; c < 64; c++)
      {
        uniform &= (first == op(page, c));
      }
    return uniform;
  }

  // if all ops are zero or x, return x. Else return zero
  static uint64_t unique_op_except_zero(hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *end = &page->cacheline[hostrpc::page_t::width];

    hostrpc::cacheline_t *line = std::find_if(
        &page->cacheline[0], end,
        [](hostrpc::cacheline_t &line) { return line.element[0] != 0; });

    if (line != end)
      {
        uint64_t op = line->element[0];
        if (std::all_of(line, end, [=](hostrpc::cacheline_t &line) {
              uint64_t e = line.element[0];
              return e != op || e != 0;
            }))
          {
            return op;
          }
      }

    return 0;
  }

  bool recur(hostrpc::page_t *page, uint32_t slot)
  {
    auto &slot_buffer = (*print_buffer)[slot];

    uint64_t op = unique_op_except_zero(page);
    if (op == 0)
      {
        // failure time
        printf("invalid recur\n");
        return true;
      }

    if (op == func_piecewise_print_end)
      {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"
        for (unsigned c = 0; c < 64; c++)
          {
            auto &thread_print = slot_buffer[c];
            size_t N = thread_print.args_.size();

            switch (N)
              {
                case 0:
                  {
                    thread_print.acc.append_cstr<8>("(null)\n");
                    thread_print.append_acc();
                    printf(reinterpret_cast<const char *>(thread_print(0)), c);
                    break;
                  }
                case 1:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c);
                    break;
                  }
                case 2:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c,
                           thread_print(1));
                    break;
                  }
                case 3:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c,
                           thread_print(1), thread_print(2));
                    break;
                  }

                case 4:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c,
                           thread_print(1), thread_print(2), thread_print(3));
                    break;
                  }
                case 5:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c,
                           thread_print(1), thread_print(2), thread_print(3),
                           thread_print(4));
                    break;
                  }
                case 6:
                  {
                    printf(reinterpret_cast<const char *>(thread_print(0)), c,
                           thread_print(1), thread_print(2), thread_print(3),
                           thread_print(4), thread_print(5));
                    break;
                  }

                default:
                  {
                    printf("[%.2u] %s took %lu args\n", c,
                           reinterpret_cast<const char *>(thread_print(0)),
                           N - 1);
                    break;
                  }
              }
          }
#pragma clang diagnostic pop
        return true;
      }

    if (op == func_piecewise_pass_element_cstr)
      {
        for (unsigned c = 0; c < 64; c++)
          {
            auto &thread_print = slot_buffer[c];
            hostrpc::cacheline_t *line = &page->cacheline[c];
            if (line->element[0] == func_print_nop)
              {
                continue;
              }

            piecewise_pass_element_cstr_t *p =
                reinterpret_cast<piecewise_pass_element_cstr_t *>(
                    &line->element[0]);

            if (thread_print.acc.tag == func_print_nop)
              {
                thread_print.acc = print_wip::field::cstr();
              }

            if (thread_print.acc.tag == func_piecewise_pass_element_cstr)
              {
                thread_print.acc
                    .append_cstr<piecewise_pass_element_cstr_t::width>(
                        p->payload);

                if (fs_contains_nul(p->payload,
                                    piecewise_pass_element_cstr_t::width))
                  {
                    // end of string
                    thread_print.append_acc();
                  }
              }
            else
              {
                printf("invalid print cstr\n");
                return true;
              }
          }
        return false;
      }

    if (op == func_piecewise_pass_element_scalar)
      {
        for (unsigned c = 0; c < 64; c++)
          {
            auto &thread_print = slot_buffer[c];
            hostrpc::cacheline_t *line = &page->cacheline[c];
            if (line->element[0] == func_print_nop)
              {
                continue;
              }

            piecewise_pass_element_scalar_t *p =
                reinterpret_cast<piecewise_pass_element_scalar_t *>(
                    &line->element[0]);

            switch (p->Type)
              {
                case func_piecewise_pass_element_uint64:
                  {
                    if (thread_print.acc.tag == func_print_nop)
                      {
                        thread_print.acc = {p->payload};
                        thread_print.append_acc();
                      }
                    else
                      {
                        printf("invalid pass u64\n");
                      }
                    break;
                  }
                default:
                  {
                    printf("unimplemented scalar element: %lu\n", p->Type);
                    break;
                  }
              }
          }
        return false;
      }

    printf("Nested operate got work to do\n");
    return false;
  }

  void operator()(hostrpc::page_t *page)
  {
    const bool verbose = false;
    uint32_t slot = page - start_local_buffer;
    if (verbose) fprintf(stderr, "Invoked operate on slot %u\n", slot);

    auto &slot_buffer = (*buffer)[slot];
    auto &print_slot_buffer = (*print_buffer)[slot];

    if (uint64_t o = unique_op_except_zero(page))
      {
        if (o == func_piecewise_print_start)
          {
            for (unsigned c = 0; c < 64; c++)
              {
                auto &thread_print = print_slot_buffer[c];
                thread_print.clear();
                thread_print.acc = print_wip::field::cstr();
                thread_print.acc.append_cstr<7>("[%.2u] ");
              }

            // Need to conclude this current packet before recursing
            // haven't written anything, so probably no point calling
            // push from server too client, but need to tell the client
            // this packet has been recieved before the recursive call
            // can work

            // Claim this operate call has returned

            // Will wait, probably need to clear the packet

            bool got_end_packet = false;

            // The transition is roughly
            // 1/ wait until available
            // 2/ operate given available
            // 3/ publish result
            // At this point in the control flow, we are at 2/
            // Further packets can be processed with 3->1->2
            // Then drop out of this function to pick up publish

            while (!got_end_packet)
              {
                // Close this operation
                ThisServer->rpc_port_operate_publish_operate_done(slot);

                // Wait for next task
                ThisServer->rpc_port_wait_until_available(slot);

                // Do next task, but don't publish the result
                ThisServer->rpc_port_operate_given_available_nopublish(
                    [&](hostrpc::page_t *page) {
                      got_end_packet = recur(page, slot);
                    },
                    slot);
              }

            return;
          }
      }

    for (unsigned c = 0; c < 64; c++)
      {
        auto &thread_buffer = slot_buffer[c];
        hostrpc::cacheline_t *line = &page->cacheline[c];

        uint64_t ID = op(page, c);

        switch (ID)
          {
            case 0:
            default:
              {
                break;
              }

            case func_print_start:
              {
                if (verbose)
                  printf("[%.2u] got print_start, clear slot %u\n", c, slot);
                thread_buffer = {};
                break;
              }

            case func_print_finish:
              {
                print_finish i(
                    reinterpret_cast<const char *>(&line->element[0]));

                if (verbose)
                  printf(
                      "[%.2u] got print_finish on slot %u, packets %lu, buffer "
                      "size %zu\n",
                      c, slot, i.packets, thread_buffer.size());

                std::string acc;

                for (size_t p = 0; p < i.packets; p++)
                  {
                    acc.append(thread_buffer[p]);
                    if (verbose)
                      printf("[%.2u] part [%zu]: %s\n", c, p,
                             thread_buffer[p].c_str());
                  }

                printf("[%.2u] %s\n", c, acc.c_str());

                break;
              }

            case func_print_append_str:
              {
                print_append_str i(
                    reinterpret_cast<const char *>(&line->element[0]));
                auto &start_thread_buffer = (*buffer)[i.start_port][c];
                // this constructor can lead to embedded nul + noise, but
                // finish using c_str means said nul/noise is ignored
                start_thread_buffer[i.position] =
                    std::string(i.payload, sizeof(i.payload));
                if (verbose)
                  printf("[%.2u] (*buffer-%p)[%lu][%u][%lu] (sz %zu) = %s\n", c,
                         buffer, i.start_port, c, i.position,
                         start_thread_buffer.size(),
                         start_thread_buffer[i.position].c_str());

                break;
              }

            case func_print_uuu:
              {
                print_uuu_instance i(
                    reinterpret_cast<const char *>(&line->element[0]));

                constexpr unsigned pre = 7;
                const char *prefix = "[%.2u] ";

                char fmt[pre + sizeof(print_uuu_instance::fmt) + 1];
                __builtin_memcpy(&fmt, prefix, pre);
                __builtin_memcpy(&fmt[pre], i.fmt, sizeof(i.fmt));
                fmt[sizeof(fmt) - 1] = '\0';
                printf(fmt, c, line->element[5], line->element[6],
                       line->element[7]);
                break;
              }

            case func_piecewise_print_start:
              {
                printf(
                    "[%.2u] Non-uniform func_piecewise_print_start "
                    "unsupported\n",
                    c);

                break;
              }
          }
      }

    (void)page;
  }
};

struct clear
{
  void operator()(hostrpc::page_t *page)
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

using sts_ty = hostrpc::server_thread_state<
    hostrpc::x64_gcn_type<SZ>::server_type,
    operate<hostrpc::x64_gcn_type<SZ>::server_type>, clear>;

struct wrap_state
{
  std::unique_ptr<hostrpc::x64_gcn_type<SZ> > p;
  HOSTRPC_ATOMIC(uint32_t) server_control;

  sts_ty server_state;
  std::unique_ptr<hostrpc::thread<sts_ty> > thrd;

  std::unique_ptr<buffer_t> buffer;
  std::unique_ptr<print_buffer_t> print_buffer;

  wrap_state(wrap_state &&) = delete;
  wrap_state &operator=(wrap_state &&) = delete;

  wrap_state(hsa_agent_t kernel_agent)
  {
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);

    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }

    uint32_t nonblocking_size;
    {
      uint32_t cus = hsa::agent_get_info_compute_unit_count(kernel_agent);
      uint32_t waves = hsa::agent_get_info_max_waves_per_cu(kernel_agent);
      nonblocking_size = cus * waves;
    }

    SZ N{nonblocking_size};

    // having trouble getting clang to call the move constructor, work around
    // with hoeap
    p = std::make_unique<hostrpc::x64_gcn_type<SZ> >(
        N, fine_grained_region.handle, coarse_grained_region.handle);
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 1);

    buffer = std::make_unique<buffer_t>();
    print_buffer = std::make_unique<print_buffer_t>();
    print_buffer->resize(N.N());

    operate<hostrpc::x64_gcn_type<SZ>::server_type> op(
        buffer.get(), print_buffer.get(), &p->server);
    server_state = sts_ty(&p->server, &server_control, op, clear{});

    thrd =
        std::make_unique<hostrpc::thread<sts_ty> >(make_thread(&server_state));

    if (!thrd->valid())
      {
        fprintf(stderr, "Failed to spawn thread\n");
        exit(1);
      }
  }

  ~wrap_state()
  {
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
        &server_control, 0);

    thrd->join();
  }
};

struct global
{
  hsa::init hsa_instance;
  std::vector<std::unique_ptr<wrap_state> > state;
  static pthread_mutex_t mutex;

  global(const global &) = delete;
  global(global &&) = delete;

  global() : hsa_instance{} {}

  hostrpc::x64_gcn_type<SZ> *spawn(hsa_agent_t kernel_agent)
  {
    lock l(&mutex);
    state.emplace_back(
        std::unique_ptr<wrap_state>(new wrap_state(kernel_agent)));
    hostrpc::x64_gcn_type<SZ> *r = state.back().get()->p.get();
    return r;
  }

 private:
  struct lock
  {
    lock(pthread_mutex_t *m) : m(m) { pthread_mutex_lock(m); }
    ~lock() { pthread_mutex_unlock(m); }
    pthread_mutex_t *m;
  };
} global_instance;
pthread_mutex_t global::mutex = PTHREAD_MUTEX_INITIALIZER;
}  // namespace

int hostrpc::print_enable(hsa_executable_t ex, hsa_agent_t kernel_agent)
{
  const char *gpu_local_ptr = "hostrpc_x64_gcn_debug_client";

  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc = hsa_executable_get_symbol_by_name(ex, gpu_local_ptr,
                                                        &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        return 1;
      }

    hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
    if (kind != HSA_SYMBOL_KIND_VARIABLE)
      {
        return 1;
      }
  }

  void *addr =
      reinterpret_cast<void *>(hsa::symbol_get_info_variable_address(symbol));

  hostrpc::x64_gcn_type<SZ> *p = global_instance.spawn(kernel_agent);

  for (uint64_t i = 0; i < p->server.size(); i++)
    {
      hostrpc::page_t *page = &p->server.local_buffer[i];
      clear()(page);
    }

  {
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    auto c = hsa::allocate(fine_grained_region, sizeof(p->client));
    void *vc = c.get();
    memcpy(vc, &p->client, sizeof(p->client));
    int rc = hsa::copy_host_to_gpu(kernel_agent, addr, vc, sizeof(p->client));
    if (rc != 0)
      {
        return 1;
      }
  }

  return 0;
}

#endif

#if (HOSTRPC_AMDGCN)
extern "C" void example(void)
{
  unsigned id = platform::get_lane_id();

  uint32_t port = piecewise_print_start(
      "some format %u too "
      "loffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
      "ffffffffffffffffffffffffffffng with %s fields\n");
  piecewise_pass_element_uint64(port, 42);
  piecewise_pass_element_cstr(port, "stringy");
  piecewise_print_end(port);

  if (id % 3 == 0)
    {
      print_string(
          "this string is somewhat too long to fit in a single buffer so "
          "splits across several\n");
    }
  else
    {
      print_string("mostly short though\n");
    }

  platform::sleep();

  hostrpc::print("test %lu call\n", 42, 0, 0);

  if (id % 2) hostrpc::print("second %lu/%lu/%lu call\n", 101, 5, 2);
}

#endif

#if (HOSTRPC_HOST)
// tests

INCBIN(x64_gcn_debug_so, "x64_gcn_debug.gcn.so");

int main()
{
  std::vector<hsa_agent_t> gpus = hsa::find_gpus();
  for (auto g : gpus)
    {
      hsa_queue_t *queue = hsa::create_queue(g);
      auto ex =
          hsa::executable(g, x64_gcn_debug_so_data, x64_gcn_debug_so_size);
      if (!ex.valid())
        {
          printf("gpu %lu ex not valid\n", g.handle);
          return 1;
        }

      int rc = hostrpc::print_enable(ex, g);
      if (rc != 0)
        {
          printf("gpu %lu, enable -> %u\n", g.handle, rc);
          return 1;
        }

      hsa_signal_t sig;
      if (hsa_signal_create(1, 0, NULL, &sig) != HSA_STATUS_SUCCESS)
        {
          return 1;
        }

      if (hsa::launch_kernel(ex, queue, "__device_example.kd", 0, 0, sig) != 0)
        {
          return 1;
        }

      do
        {
        }
      while (hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_EQ, 0,
                                     5000 /*000000*/,
                                     HSA_WAIT_STATE_ACTIVE) != 0);

      hsa_signal_destroy(sig);
      hsa_queue_destroy(queue);

      return 0;  // skip the second gpu
    }
}

#endif

#endif
