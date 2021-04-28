// This includes platform which contains a stub for printf
#include "x64_gcn_type.hpp"
#undef printf

#include "hostrpc_printf.h"

#include "cxa_atexit.hpp"

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

  HOSTRPC_ANNOTATE print_uuu_instance(const char *fmt_, uint64_t x0,
                                      uint64_t x1, uint64_t x2)
      : arg0(x0), arg1(x1), arg2(x2)
  {
    __builtin_memcpy(fmt, fmt_, fs_strnlen(fmt_, sizeof(fmt)));
  }

  HOSTRPC_ANNOTATE print_uuu_instance(const char *d)
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
  HOSTRPC_ANNOTATE print_start() {}
};
struct print_finish
{
  uint64_t ID = func_print_finish;
  uint64_t packets;
  char unused[48];
  HOSTRPC_ANNOTATE print_finish(uint64_t packets) : packets(packets) {}

  HOSTRPC_ANNOTATE print_finish(const char *d)
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
  HOSTRPC_ANNOTATE print_append_str(uint64_t start_port, uint64_t position,
                                    const char *str)
      : start_port(start_port), position(position)
  {
    __builtin_memcpy(payload, str, fs_strnlen(str, sizeof(payload)));
  }

  HOSTRPC_ANNOTATE print_append_str(const char *d)
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
  HOSTRPC_ANNOTATE piecewise_print_start_t() {}
};

struct piecewise_print_end_t
{
  uint64_t ID = func_piecewise_print_end;
  char unused[56];
  HOSTRPC_ANNOTATE piecewise_print_end_t() {}
};

struct piecewise_pass_element_cstr_t
{
  uint64_t ID = func_piecewise_pass_element_cstr;
  enum
  {
    width = 56
  };
  char payload[width];
  HOSTRPC_ANNOTATE piecewise_pass_element_cstr_t(const char *s, size_t N)
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
  HOSTRPC_ANNOTATE piecewise_pass_element_scalar_t(enum func_type type,
                                                   uint64_t x)
      : Type(type), payload(x)
  {
  }
};

using SZ = hostrpc::size_runtime;

}  // namespace

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<SZ>::client_type hostrpc_x64_gcn_debug_client[1];

template <typename T>
struct fill_by_copy
{
  static_assert(sizeof(T) == 64, "");
  HOSTRPC_ANNOTATE fill_by_copy(T *i) : i(i) {}
  T *i;

  HOSTRPC_ANNOTATE void operator()(hostrpc::page_t *page)
  {
    unsigned id = platform::get_lane_id();
    hostrpc::cacheline_t *dline = &page->cacheline[id];
    __builtin_memcpy(dline, i, 64);
  }
};

__PRINTF_API_EXTERNAL uint32_t piecewise_print_start(const char *fmt)
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

__PRINTF_API_EXTERNAL int piecewise_print_end(uint32_t port)
{
  {
    piecewise_print_end_t inst;
    fill_by_copy<piecewise_print_end_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  hostrpc_x64_gcn_debug_client[0].rpc_close_port(port);
  return 0;  // should be return code from printf
}

// These may want to be their own functions, for now delagate to u64
__PRINTF_API_EXTERNAL void piecewise_pass_element_int32(uint32_t port,
                                                        int32_t v)
{
  int64_t w = v;
  return piecewise_pass_element_int64(port, w);
}

__PRINTF_API_EXTERNAL void piecewise_pass_element_uint32(uint32_t port,
                                                         uint32_t v)
{
  uint64_t w = v;
  return piecewise_pass_element_uint64(port, w);
}

__PRINTF_API_EXTERNAL void piecewise_pass_element_int64(uint32_t port,
                                                        int64_t v)
{
  uint64_t c;
  __builtin_memcpy(&c, &v, 8);
  return piecewise_pass_element_uint64(port, c);
}

__PRINTF_API_EXTERNAL void piecewise_pass_element_uint64(uint32_t port,
                                                         uint64_t v)
{
  piecewise_pass_element_scalar_t inst(func_piecewise_pass_element_uint64, v);
  fill_by_copy<piecewise_pass_element_scalar_t> f(&inst);
  hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
}

__PRINTF_API_EXTERNAL void piecewise_pass_element_cstr(uint32_t port,
                                                       const char *str)
{
  uint64_t L = __printf_strlen(str);

  const constexpr size_t w = piecewise_pass_element_cstr_t::width;

  // this appears to behave poorly when different threads make different numbers
  // of calls
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

using print_buffer_t = std::vector<std::array<print_wip, 64> >;

template <typename ServerType>
struct operate
{
  print_buffer_t *print_buffer = nullptr;
  ServerType *ThisServer;
  hostrpc::page_t *start_local_buffer = nullptr;
  operate(print_buffer_t *print_buffer, ServerType *ThisServer)
      : print_buffer(print_buffer), ThisServer(ThisServer)
  {
    start_local_buffer = ThisServer->local_buffer;
  }
  operate() = default;

  void doprint(unsigned c, print_wip &thread_print)
  {
    size_t N = thread_print.args_.size();
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"

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
                   reinterpret_cast<const char *>(thread_print(0)), N - 1);
            break;
          }
      }
#pragma clang diagnostic pop
  }

  void perthread(unsigned c, hostrpc::cacheline_t *line,
                 print_wip &thread_print)
  {
    uint64_t ID = line->element[0];

    switch (ID)
      {
        case func_print_nop:
          {
            break;
          }

        case func_piecewise_print_start:
          {
            thread_print.clear();
            thread_print.acc = print_wip::field::cstr();
            thread_print.acc.append_cstr<7>("[%.2u] ");
            break;
          }

        case func_piecewise_print_end:
          {
            doprint(c, thread_print);
            break;
          }

        case func_piecewise_pass_element_cstr:
          {
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
              }

            break;
          }

        case func_piecewise_pass_element_scalar:
          {
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

            break;
          }

        default:
          {
            printf("Unhandled op: %lu\n", ID);
          }
      }
  }

  void operator()(hostrpc::page_t *page)
  {
    const bool verbose = false;
    uint32_t slot = page - start_local_buffer;
    if (verbose) fprintf(stderr, "Invoked operate on slot %u\n", slot);

    std::array<print_wip, 64> &print_slot_buffer = (*print_buffer)[slot];

    for (unsigned c = 0; c < 64; c++)
      {
        perthread(c, &page->cacheline[c], print_slot_buffer[c]);
      }
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

struct global
{
  struct wrap_state
  {
    using sts_ty = hostrpc::server_thread_state<
        hostrpc::x64_gcn_type<SZ>::server_type,
        operate<hostrpc::x64_gcn_type<SZ>::server_type>, clear>;

    std::unique_ptr<hostrpc::x64_gcn_type<SZ> > p;
    HOSTRPC_ATOMIC(uint32_t) server_control;

    sts_ty server_state;
    std::unique_ptr<hostrpc::thread<sts_ty> > thrd;

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
      // with heap
      p = std::make_unique<hostrpc::x64_gcn_type<SZ> >(
          N, fine_grained_region.handle, coarse_grained_region.handle);
      platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                             __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          &server_control, 1);

      print_buffer = std::make_unique<print_buffer_t>();
      print_buffer->resize(N.N());

      operate<hostrpc::x64_gcn_type<SZ>::server_type> op(print_buffer.get(),
                                                         &p->server);
      server_state = sts_ty(&p->server, &server_control, op, clear{});

      thrd = std::make_unique<hostrpc::thread<sts_ty> >(
          make_thread(&server_state));

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

int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                      hsa_agent_t kernel_agent)
{
  const bool verbose = false;
  if (verbose) fprintf(stderr, "called print enable\n");
  const char *gpu_local_ptr = "hostrpc_x64_gcn_debug_client";

  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc = hsa_executable_get_symbol_by_name(ex, gpu_local_ptr,
                                                        &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        if (verbose) fprintf(stderr, "can't find symbool %s\n", gpu_local_ptr);
        return 1;
      }

    hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
    if (kind != HSA_SYMBOL_KIND_VARIABLE)
      {
        if (verbose) fprintf(stderr, "symbol not a variable\n");
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
    if (verbose)
      if (!vc) fprintf(stderr, "Alloc failed\n");
    memcpy(vc, &p->client, sizeof(p->client));
    int rc = hsa::copy_host_to_gpu(kernel_agent, addr, vc, sizeof(p->client));
    if (rc != 0)
      {
        if (verbose) fprintf(stderr, "Copy host to gpu failed\n");
        return 1;
      }
  }

  if (verbose) fprintf(stderr, "Returning success (0) from print_enable\n");
  return 0;
}

#endif