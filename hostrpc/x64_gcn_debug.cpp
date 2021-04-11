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

enum : uint64_t
{
  func_print_nop = 0,
  func_print_uuu = 1,
  func_print_start = 2,
  func_print_finish = 3,
  func_print_append_str = 4,

  func_debug_print_start = 5,
  func_debug_print_end = 5,

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

struct debug_print_start_t
{
  uint64_t ID = func_debug_print_start;
  char unused[56];
  debug_print_start_t() {}
};

struct debug_print_end_t
{
  uint64_t ID = func_debug_print_end;
  char unused[56];
  debug_print_end_t() {}
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
  size_t N = fs_strlen(str);
  (void)N;
  // Get a port
  uint32_t port = hostrpc_x64_gcn_debug_client[0].rpc_open_port();
  if (port == UINT32_MAX)
    {
      // failure
      return;
    }

  // Start a transaction
  {
    print_start inst;
    fill_by_copy<print_start> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(
        port, f);  // require f() to have been called before this return
  }

  uint64_t pos = 0;
  uint64_t length = fs_strlen(str);
  constexpr size_t chunk = sizeof(print_append_str::payload);
  uint64_t pieces = (length + chunk - 1) / chunk;

  // Append the string, in pieces, via various ports
  // Probably want try_invoke or similar, don't need to
  // fill the whole buffer either
  // Also, there may be zero available, in which case some can be
  // sent down the existing pipe
  for (uint64_t i = 0; i < pieces; i++)
    {
      print_append_str inst(port, pos++, &str[i * chunk]);
      fill_by_copy<print_append_str> f(&inst);
      hostrpc_x64_gcn_debug_client[0].rpc_invoke_async(f);
    }

  // Emit the string, using the original port. Will therefore
  // execute after the print_start
  // TODO: Also need to wait for the pieces to land, probably on
  // the gpu side. Needs an invoke_async that returns a handle
  // that can be waited on or similar, or for the host to wait
  // until the 'pos' packets have arrived. That's tricky without
  // coroutines or similar on the server, as once the finish has
  // started there may be no thread handling the pieces
  // Or send them all down the same pipe, simpler but expected
  // slower
  {
    print_finish inst(pos);
    fill_by_copy<print_finish> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  // Wait for the above to flush before returning from this call
  {
    // wait until the server process has run the function
    // but don't need too wait until the slot is available again
    // as it's about to close anyway
    hostrpc_x64_gcn_debug_client[0].rpc_port_wait_for_result(port);
  }

  // Clean up
  hostrpc_x64_gcn_debug_client[0].rpc_close_port(port);
}

#if 0
// server cases are much easier to write if one port is used
// for multiple packets
struct print_wip
{
  std::vector<char> format;

  struct field
  {
    uint64_t tag;
    union {
      uint64_t u64;
      std::vector<char> cstr;
    };
    
  };

  std::vector<field> args;
};
#endif

void debug_print_pass_cstr(uint32_t port, const char *);  // used to pass fmt

uint32_t debug_print_start(const char *fmt)
{
  uint32_t port = hostrpc_x64_gcn_debug_client[0].rpc_open_port();
  if (port == UINT32_MAX)
    {
      // failure
      UINT32_MAX;
    }

  {
    debug_print_start_t inst;
    fill_by_copy<debug_print_start_t> f(&inst);
    hostrpc_x64_gcn_debug_client[0].rpc_port_send(port, f);
  }

  debug_print_pass_cstr(port, fmt);

  return port;
}

void debug_print_pass_uint64(uint32_t port, uint64_t);

void debug_print_pass_cstr(uint32_t port, const char *) { (void)port; }

int debug_print_end(uint32_t port);

#endif

#if (HOSTRPC_HOST)

#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "incbin.h"
#include "server_thread_state.hpp"

#include <pthread.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
using buffer_t = std::unordered_map<
    uint32_t, std::array<std::unordered_map<uint64_t, std::string>, 64> >;

struct operate
{
  buffer_t *buffer = nullptr;
  hostrpc::page_t *start_local_buffer = nullptr;
  operate(buffer_t *buffer, hostrpc::page_t *s)
      : buffer(buffer), start_local_buffer(s)
  {
  }
  operate() = default;

  void operator()(hostrpc::page_t *page)
  {
    const bool verbose = false;
    uint32_t slot = page - start_local_buffer;
    if (verbose) fprintf(stderr, "Invoked operate on slot %u\n", slot);

    auto &slot_buffer = (*buffer)[slot];

    for (unsigned c = 0; c < 64; c++)
      {
        auto &thread_buffer = slot_buffer[c];
        hostrpc::cacheline_t *line = &page->cacheline[c];

        uint64_t ID = line->element[0];

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

            case func_debug_print_start:
              {
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

using sts_ty =
    hostrpc::server_thread_state<hostrpc::x64_gcn_type<SZ>::server_type,
                                 operate, clear>;

struct wrap_state
{
  std::unique_ptr<hostrpc::x64_gcn_type<SZ> > p;
  HOSTRPC_ATOMIC(uint32_t) server_control;

  sts_ty server_state;
  std::unique_ptr<hostrpc::thread<sts_ty> > thrd;

  std::unique_ptr<buffer_t> buffer;

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

    buffer = std::make_unique<buffer_t>();

    operate op(buffer.get(), p->server.local_buffer);
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

    buffer.reset();
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

  if (id % 3 == 0)
    {
      print_string(
          "this string is somewhat too long to fit in a single buffer so "
          "splits across several");
    }
  else
    {
      print_string("mostly short though");
    }

  return;

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
