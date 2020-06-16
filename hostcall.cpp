#include "hostcall.hpp"
#include "base_types.hpp"
#include "interface.hpp"
#include "platform.hpp"

#if defined(__x86_64__)
#include "hsa.hpp"
#include <thread>
#include <vector>
#endif


static const constexpr uint32_t MAX_NUM_DOORBELLS = 0x400;

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

#if defined(__AMDGCN__)

__attribute__((visibility("default")))
hostrpc::x64_amdgcn_t::client_t client_singleton[MAX_NUM_DOORBELLS];

// Also in hsa.hpp
static uint16_t get_queue_index()
{
  static_assert(MAX_NUM_DOORBELLS < UINT16_MAX, "");
  uint32_t tmp0, tmp1;

  // Derived from mGetDoorbellId in amd_gpu_shaders.h, rocr
  // Using similar naming, exactly the same control flow.
  // This may be expensive enough to be worth caching or precomputing.
  uint32_t res;
  asm("s_mov_b32 %[tmp0], exec_lo\n\t"
      "s_mov_b32 %[tmp1], exec_hi\n\t"
      "s_mov_b32 exec_lo, 0x80000000\n\t"
      "s_sendmsg sendmsg(MSG_GET_DOORBELL)\n\t"
      "%=:\n\t"
      "s_nop 7\n\t"
      "s_bitcmp0_b32 exec_lo, 0x1F\n\t"
      "s_cbranch_scc0 %=b\n\t"
      "s_mov_b32 %[ret], exec_lo\n\t"
      "s_mov_b32 exec_lo, %[tmp0]\n\t"
      "s_mov_b32 exec_hi, %[tmp1]\n\t"
      : [ tmp0 ] "=&r"(tmp0), [ tmp1 ] "=&r"(tmp1), [ ret ] "=r"(res));

  res %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(res);
}

void hostcall_client(uint64_t data[8])
{
  hostrpc::x64_amdgcn_t::client_t &c = client_singleton[get_queue_index()];

  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = c.invoke(d, d);
    }
}

void hostcall_client_async(uint64_t data[8])
{
  hostrpc::x64_amdgcn_t::client_t &c = client_singleton[get_queue_index()];
  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = c.invoke_async(d, d);
    }
}

#else

// Get the start of the array
const char *hostcall_client_symbol() { return "client_singleton"; }

hostrpc::x64_amdgcn_t::server_t server_singleton[MAX_NUM_DOORBELLS];

uint16_t queue_to_index(hsa_queue_t *q)
{
  char *sig = reinterpret_cast<char *>(q->doorbell_signal.handle);
  int64_t kind;
  __builtin_memcpy(&kind, sig, 8);
  // TODO: Work out if any hardware that works for openmp uses legacy doorbell
  assert(kind == -1);
  sig += 8;

  const uint64_t MAX_NUM_DOORBELLS = 0x400;

  uint64_t ptr;
  __builtin_memcpy(&ptr, sig, 8);
  ptr >>= 3;
  ptr %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(ptr);
}

hostrpc::x64_amdgcn_t *stored_pairs[MAX_NUM_DOORBELLS] = {0};

#if defined(__x86_64__)

class hostcall_impl
{
 public:
  hostcall_impl(hsa_executable_t executable, hsa_agent_t kernel_agent);

  hostcall_impl(hostcall_impl &&o)
      : clients(std::move(o.clients)),
        servers(std::move(o.servers)),
        stored_pairs(std::move(o.stored_pairs)),
        queue_loc(std::move(o.queue_loc)),
        threads(std::move(o.threads)),
        fine_grained_region(std::move(o.fine_grained_region)),
        coarse_grained_region(std::move(o.coarse_grained_region))
  {
    clients = 0;
    servers = {};
    stored_pairs = {};
    queue_loc = {};
    // threads = {};
  }

  hostcall_impl(const hostcall_impl &) = delete;

  static uint64_t find_symbol_address(hsa_executable_t &ex,
                                      hsa_agent_t kernel_agent,
                                      const char *sym);

  int enable_queue(hsa_queue_t *queue)
  {
    uint16_t queue_id = queue_to_index(queue);
    assert(stored_pairs[queue_id] == 0);

    // TODO: Avoid this heap alloc
    hostrpc::x64_amdgcn_t *res = new hostrpc::x64_amdgcn_t(
        fine_grained_region.handle, coarse_grained_region.handle);
    if (!res)
      {
        return 1;
      }

    clients[queue_id] = res->client();

    servers[queue_id] = res->server();

    stored_pairs[queue_id] = res;

    return 0;
  }

  int spawn_worker(hsa_queue_t *queue)
  {
    uint16_t queue_id = queue_to_index(queue);
    if (stored_pairs[queue_id] == 0)
      {
        return 1;
      }
    return spawn_worker(queue_id);
  }

  ~hostcall_impl()
  {
    for (size_t i = 0; i < MAX_NUM_DOORBELLS; i++)
      {
        delete stored_pairs[i];
      }
    thread_killer = 1;
    for (size_t i = 0; i < threads.size(); i++)
      {
        threads[i].join();
      }
  }

 private:
  int spawn_worker(uint16_t queue_id)
  {
    _Atomic uint32_t *control = &thread_killer;
    hostrpc::x64_amdgcn_t::server_t *server = &servers[queue_id];
    uint64_t *ql = &queue_loc[queue_id];
    // TODO. Can't actually use std::thread because the constructor throws.
    threads.emplace_back([control, server, ql]() {
      for (;;)
        {
          while (server->handle(nullptr, ql))
            {
            }

          if (*control != 0)
            {
              return;
            }

          // yield
        }
    });
    return 0;  // can't detect errors from std::thread
  }

  // Going to need these to be opaque
  hostrpc::x64_amdgcn_t::client_t *clients;  // statically allocated

  // heap allocated, may not need the servers() instance
  std::vector<hostrpc::x64_amdgcn_t::server_t> servers;
  std::vector<hostrpc::x64_amdgcn_t *> stored_pairs;
  std::vector<uint64_t> queue_loc;

  _Atomic uint32_t thread_killer = 0;
  std::vector<std::thread> threads;
  hsa_region_t fine_grained_region;
  hsa_region_t coarse_grained_region;
};

// todo: port to hsa.h api

hostcall_impl::hostcall_impl(hsa_executable_t executable,
                             hsa_agent_t kernel_agent)
{
  // The client_t array is per-gpu-image. Find it.
  uint64_t client_addr =
      find_symbol_address(executable, kernel_agent, hostcall_client_symbol());
  clients = reinterpret_cast<hostrpc::x64_amdgcn_t::client_t *>(client_addr);

  // todo: error checks here
  fine_grained_region = hsa::region_fine_grained(kernel_agent);

  bool faster = false;  // coarse grain on locks isn't helping at present
  coarse_grained_region =
      faster ? hsa::region_coarse_grained(kernel_agent) : fine_grained_region;

  // probably can't use vector for exception-safety reasons
  servers.resize(MAX_NUM_DOORBELLS);
  stored_pairs.resize(MAX_NUM_DOORBELLS);
  queue_loc.resize(MAX_NUM_DOORBELLS);
}

uint64_t hostcall_impl::find_symbol_address(hsa_executable_t &ex,
                                            hsa_agent_t kernel_agent,
                                            const char *sym)
{
  // TODO: This was copied from the loader, sort out the error handling
  hsa_executable_symbol_t symbol;
  {
    hsa_status_t rc =
        hsa_executable_get_symbol_by_name(ex, sym, &kernel_agent, &symbol);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "HSA failed to find symbol %s\n", sym);
        exit(1);
      }
  }

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
  if (kind != HSA_SYMBOL_KIND_VARIABLE)
    {
      fprintf(stderr, "Symbol %s is not a variable\n", sym);
      exit(1);
    }

  return hsa::symbol_get_info_variable_address(symbol);
}

template <size_t expect, size_t actual>
static void assert_size_t_equal()
{
  static_assert(expect == actual, "");
}

hostcall::hostcall(hsa_executable_t executable, hsa_agent_t kernel_agent)
{
  assert_size_t_equal<hostcall::state_t::align(), alignof(hostcall_impl)>();
  assert_size_t_equal<hostcall::state_t::size(), sizeof(hostcall_impl)>();
  new (reinterpret_cast<hostcall_impl *>(state.data))
      hostcall_impl(hostcall_impl(executable, kernel_agent));
}

bool hostcall::valid() { return true; }

int hostcall::enable_queue(hsa_queue_t *queue)
{
  return state.open<hostcall_impl>()->enable_queue(queue);
}
int hostcall::spawn_worker(hsa_queue_t *queue)
{
  return state.open<hostcall_impl>()->spawn_worker(queue);
}

#endif

#endif
