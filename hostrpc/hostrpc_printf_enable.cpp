#include "hostrpc_printf_enable.h"
#include "hostrpc_printf.h"

#include "x64_gcn_type.hpp"
#undef printf

#include "cxa_atexit.hpp"

#include "hostrpc_printf_server.hpp"

#if (HOSTRPC_AMDGCN)

__attribute__((visibility("default")))
hostrpc::x64_gcn_type<hostrpc::size_runtime>::client_type
    hostrpc_x64_gcn_debug_client[1];

__PRINTF_API_EXTERNAL uint32_t __printf_print_start(const char *fmt)
{
  return __printf_print_start(&hostrpc_x64_gcn_debug_client[0], fmt);
}

__PRINTF_API_EXTERNAL int __printf_print_end(uint32_t port)
{
  return __printf_print_end(&hostrpc_x64_gcn_debug_client[0], port);
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
  return __printf_pass_element_uint64(&hostrpc_x64_gcn_debug_client[0], port,
                                      v);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_double(uint32_t port, double v)
{
  return __printf_pass_element_double(&hostrpc_x64_gcn_debug_client[0], port,
                                      v);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_void(uint32_t port,
                                                      const void *v)
{
  __printf_pass_element_void(&hostrpc_x64_gcn_debug_client[0], port, v);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_cstr(uint32_t port,
                                                      const char *str)
{
  __printf_pass_element_cstr(&hostrpc_x64_gcn_debug_client[0], port, str);
}

__PRINTF_API_EXTERNAL void __printf_pass_element_write_int64(uint32_t port,
                                                             int64_t *x)
{
  __printf_pass_element_write_int64(&hostrpc_x64_gcn_debug_client[0], port, x);
}

#endif

#if (HOSTRPC_HOST)

#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "server_thread_state.hpp"

#include "incprintf.hpp"

#include <algorithm>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{

using SZ = hostrpc::size_runtime;

struct global
{
  struct wrap_state
  {
    using sts_ty =
        hostrpc::server_thread_state<hostrpc::x64_gcn_type<SZ>::server_type,
                                     operate, clear>;

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

      operate op(print_buffer.get());
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
  const bool verbose = true;

  if (verbose) fprintf(stderr, "called print enable\n");
  const char *gpu_local_ptr = "hostrpc_x64_gcn_debug_client";

  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc = hsa_executable_get_symbol_by_name(ex, gpu_local_ptr,
                                                        &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        if (verbose) fprintf(stderr, "can't find symbol %s\n", gpu_local_ptr);
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

  for (uint32_t i = 0; i < p->server.size(); i++)
    {
      hostrpc::page_t *page = &p->server.shared_buffer[i];
      clear()(i, page);
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
