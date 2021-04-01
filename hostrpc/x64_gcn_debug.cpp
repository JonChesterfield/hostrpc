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
struct print_instance
{
  uint64_t ID = 0;
  char fmt[32] = {0};
  uint64_t arg0 = 0;
  uint64_t arg1 = 0;
  uint64_t arg2 = 0;

  print_instance() = default;
  print_instance(const char *d)
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

static constexpr unsigned debug_print_chars()
{
  return sizeof(print_instance::fmt);
}

using SZ = hostrpc::size_runtime;

}  // namespace

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<SZ>::client_type hostrpc_x64_gcn_debug_client[1];

static unsigned count_chars(const char *str)
{
  for (unsigned i = 0; i < debug_print_chars(); i++)
    {
      if (str[i] == '\0')
        {
          return i;
        }
    }
  return debug_print_chars();
}

void hostrpc::print_base(const char *str, uint64_t x0, uint64_t x1, uint64_t x2)
{
  struct fill
  {
    fill(print_instance *i) : i(i) {}
    print_instance *i;

    void operator()(hostrpc::page_t *page)
    {
      unsigned id = platform::get_lane_id();
      hostrpc::cacheline_t *dline = &page->cacheline[id];
      __builtin_memcpy(dline, i, 64);
    };
  };

  struct use
  {
    void operator()(hostrpc::page_t *){};
  };

  constexpr uint64_t print_base_id = 42;

  unsigned N = count_chars(str);

  print_instance i;
  i.ID = print_base_id;
  i.arg0 = x0;
  i.arg1 = x1;
  i.arg2 = x2;

  __builtin_memcpy(i.fmt, str, N);

  fill f(&i);
  use u;
  hostrpc_x64_gcn_debug_client[0].rpc_invoke<fill, use>(f, u);
}

#endif

#if (HOSTRPC_HOST)

#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "incbin.h"
#include "server_thread_state.hpp"

#include <pthread.h>
#include <stdio.h>
#include <utility>
#include <vector>

namespace
{
struct operate
{
  void operator()(hostrpc::page_t *page)
  {
    constexpr unsigned pre = 7;
    const char *prefix = "[%.2u] ";

    char fmt[pre + debug_print_chars() + 1];
    __builtin_memcpy(&fmt, prefix, pre);

    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t *line = &page->cacheline[c];
        print_instance i(reinterpret_cast<const char *>(&line->element[0]));

        memcpy(&fmt[pre], i.fmt, sizeof(i.fmt));
        fmt[sizeof(fmt) - 1] = '\0';
        printf(fmt, c, line->element[5], line->element[6], line->element[7]);
      }

    (void)page;
    fprintf(stderr, "Invoked operate\n");
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

    server_state = sts_ty(&p->server, &server_control, operate{}, clear{});

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
  hostrpc::print("test %lu call\n", 42, 0, 0);

  unsigned id = platform::get_lane_id();
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
    }
}

#endif

#endif
