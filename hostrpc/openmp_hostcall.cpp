#include "detail/common.hpp"
#include "hostcall.hpp"
#include "hostcall_hsa.hpp"
#include "hostrpc_printf.h"
#include "hostrpc_printf_enable.hpp"
#include "hostrpc_printf_server.hpp"
#include "platform.hpp"
#include "platform/detect.hpp"
#include "x64_gcn_type.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

enum opcodes
{
  opcodes_nop = 0,
  opcodes_malloc = 1,
  opcodes_free = 2,
};

using sizeType = hostrpc::size_runtime<uint32_t>;

#if HOSTRPC_AMDGCN
#pragma omp declare target

using client_type = hostrpc::x64_gcn_type<sizeType>::client_type;
static client_type *get_client();

struct fill
{
  uint64_t *d;
  fill(uint64_t *d) : d(d) {}
  void operator()(uint32_t, hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        line->element[i] = d[i];
      }
  }
};

struct use
{
  uint64_t *d;
  use(uint64_t *d) : d(d) {}
  void operator()(uint32_t, hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        d[i] = line->element[i];
      }
  }
};

// overrides weak functions in target_impl.hip
extern "C"
{
  void *__kmpc_impl_malloc(size_t);
  void __kmpc_impl_free(void *);
}

void *__kmpc_impl_malloc(size_t x)
{
  uint64_t data[8] = {0};
  data[0] = opcodes_malloc;
  data[1] = x;
  fill f(&data[0]);
  use u(&data[0]);
  client_type *c = get_client();
  bool success = false;
  while (!success)
    {
      success = c->rpc_invoke(f, u);
    }
  void *res;
  __builtin_memcpy(&res, &data[0], 8);
  return res;
}

void __kmpc_impl_free(void *x)
{
  uint64_t data[8] = {0};
  data[0] = opcodes_free;
  __builtin_memcpy(&data[1], &x, 8);
  fill f(&data[0]);
  client_type *c = get_client();
  bool success = false;
  while (!success)
    {
      success = c->rpc_invoke(f);  // async
    }
}

static client_type *get_client()
{
  // Less obvious is that some asm is needed on the device to trigger the
  // handling,
  __asm__(
      "; hostcall_invoke: record need for hostcall support\n\t"
      ".type needs_hostcall_buffer,@object\n\t"
      ".global needs_hostcall_buffer\n\t"
      ".comm needs_hostcall_buffer,4" ::
          :);

  // and that the result of hostrpc_assign_buffer, if zero is failure, else
  // it's written into a pointer in the implicit arguments where the GPU can
  // retrieve it from
  // size_t* argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  // result is found in argptr[3]
  // todo: does code object version change this?

  size_t *argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  return (client_type *)argptr[3];
}

#pragma omp end declare target
#endif

#if HOSTRPC_HOST

#include "hostrpc_thread.hpp"
#include "hsa.hpp"
#include "server_thread_state.hpp"

#include <list>

// overrides weak functions in rtl.cpp
extern "C"
{
  unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                      uint32_t device_id);
  hsa_status_t hostrpc_init();
  hsa_status_t hostrpc_terminate();
}

struct omp_operate
{
  hsa_region_t coarse_region;
  print_buffer_t *print_buffer = nullptr;
  omp_operate(hsa_region_t r, print_buffer_t *print_buffer)
      : coarse_region(r), print_buffer(print_buffer)
  {
  }
  void perthread(unsigned c, hostrpc::cacheline_t *line,
                 print_wip &thread_print);

  void operator()(uint32_t port, hostrpc::page_t *page)
  {
    uint32_t slot = static_cast<uint32_t>(port);
    std::array<print_wip, 64> &print_slot_buffer = (*print_buffer)[slot];
    for (unsigned c = 0; c < 64; c++)
      perthread(c, &page->cacheline[c], print_slot_buffer[c]);
  }
};

struct omp_clear
{
  void operator()(uint32_t, hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      page->cacheline[c].element[0] = opcodes_nop;
  }
};

// in a loop on a pthread,
// server->rpc_handle<operate, clear>(op, clear);

void omp_operate::perthread(unsigned c, hostrpc::cacheline_t *line,
                            print_wip &thread_print)
{
  uint64_t ID = line->element[0];
  static_assert(0 == opcodes_nop, "");
  static_assert(0 == hostrpc_printf_print_nop, "");

  bool verbose = false;

  if (operate_printf_handle(c, line, thread_print, verbose))
    {
      return;
    }

  switch (ID)
    {
      case 0:
        {
          break;
        }
      default:
        {
          printf("Unhandled ID: %lu\n", ID);
          break;
        }

      case opcodes_malloc:
        {
          uint64_t size;
          memcpy(&size, &line->element[1], 8);

          void *res;
          hsa_status_t r = hsa_memory_allocate(coarse_region, size, &res);
          if (r != HSA_STATUS_SUCCESS)
            {
              res = nullptr;
            }

          memcpy(&line->element[0], &res, 8);
          break;
        }

      case opcodes_free:
        {
          void *ptr;
          memcpy(&ptr, &line->element[1], 8);
          hsa_memory_free(ptr);
          break;
        }
    }

  return;
}

namespace
{
struct storage_t
{
  using type = hostrpc::x64_gcn_type<sizeType>;
  std::vector<type> stash;
  std::vector<type::storage_type::AllocLocal::raw> server_pointers;
  std::vector<type::storage_type::AllocRemote::raw> client_pointers;

  using sts_ty =
      hostrpc::server_thread_state<type::server_type, omp_operate, omp_clear>;

  std::list<sts_ty> thread_state;
  std::list<std::unique_ptr<print_buffer_t>> stash_print_buffer;
  std::vector<hostrpc::thread<sts_ty>> threads;

  HOSTRPC_ATOMIC(uint32_t) server_control;

  storage_t() = default;

  void drop()
  {
    size_t N = server_pointers.size();
    assert(N == client_pointers.size());
    for (size_t i = 0; i < N; i++)
      {
        // todo: ensure destroy can be called on non-valid pointers
        server_pointers[i].destroy();
        client_pointers[i].destroy();
      }
    server_pointers.clear();
    client_pointers.clear();
    stash.clear();
  }

  ~storage_t() { drop(); }
};

struct storage_global
{
  static storage_t &instance()
  {
    static storage_t storage;
    return storage;
  }

 private:
  storage_global() = default;
  storage_global(storage_global const &) = delete;
  void operator=(storage_global const &) = delete;
};
}  // namespace

hsa_status_t hostrpc_init()
{
  storage_t &storage = storage_global::instance();
  platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                         __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
      &storage.server_control, 1);

  return HSA_STATUS_SUCCESS;
}
hsa_status_t hostrpc_terminate()
{
  storage_t &storage = storage_global::instance();
  platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                         __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
      &storage.server_control, 0);

  for (auto &i : storage.threads)
    {
      i.join();
    }

  storage.drop();

  // storage = {}; // wants various copy constructors defined on storage, drop
  // it for now. todo: finish cleanup.
  return HSA_STATUS_SUCCESS;
}

unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                    uint32_t device_id)
{
  storage_t &storage = storage_global::instance();
  (void)this_Q;
  (void)device_id;

  while (storage.client_pointers.size() <= device_id)
    {
      storage.stash.push_back({});
      storage.server_pointers.push_back({});
      storage.client_pointers.push_back({});
    }

  auto as_ulong = [](uint32_t device_id) -> unsigned long {
    // assumes null is zero, todo: can that be dropped
    void *p =
        storage_global::instance().client_pointers[device_id].remote_ptr();
    unsigned long t;
    __builtin_memcpy(&t, &p, 8);
    return t;
  };

  if (unsigned long r = as_ulong(device_id))
    {
      return r;
    }

  // gets called a lot, so if the result is already available, need to return it
  // probably a vector<unsigned long>
  // is called under a lock

  uint32_t numCu;
  hsa_status_t err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &numCu);

  if (err != HSA_STATUS_SUCCESS)
    {
      return 0;
    }
  uint32_t waverPerCu;
  err = hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
      &waverPerCu);
  if (err != HSA_STATUS_SUCCESS)
    {
      return 0;
    }

  sizeType size = numCu * waverPerCu;
  hsa_region_t fine_grain = hsa::region_fine_grained(agent);
  hsa_region_t coarse_grain = hsa::region_coarse_grained(agent);

  storage.stash[device_id] =
      storage_t::type{size, {}, {fine_grain.handle, coarse_grain.handle}};

  {
    auto alloc = storage_t::type::storage_type::AllocLocal();
    storage.server_pointers[device_id] =
        alloc.allocate(sizeof(storage_t::type::server_type));
    auto local = storage.server_pointers[device_id].local_ptr();
    __builtin_memcpy(local, &storage.stash[device_id].server,
                     sizeof(storage_t::type::server_type));
  }

  // allocate pointer on gpu, memcpy client to it
  {
    constexpr size_t SZ = sizeof(storage_t::type::client_type);
    auto alloc =
        storage_t::type::storage_type::AllocRemote(coarse_grain.handle);
    storage.client_pointers[device_id] = alloc.allocate(SZ);

    auto remote = storage.client_pointers[device_id].remote_ptr();

    auto bufferS = hsa::allocate(fine_grain, SZ);
    void *buffer = bufferS.get();

    if (!buffer)
      {
        return 0;
      }
    memcpy(buffer, &storage.stash[device_id].client, SZ);

    int cp = hsa::copy_host_to_gpu(agent, remote, buffer, SZ);

    if (cp != 0)
      {
        return 0;
      }
  }

  // set up the server side printf buffer state
  storage.stash_print_buffer.emplace_back(std::make_unique<print_buffer_t>());
  std::unique_ptr<print_buffer_t> &print_buffer =
      storage.stash_print_buffer.back();
  print_buffer->resize(size.value());

  // finally spawn a thread to run the server process, and handle book keeping
  // of it
  storage.thread_state.push_back(hostrpc::make_server_thread_state(
      reinterpret_cast<storage_t::type::server_type *>(
          (void *)storage.server_pointers[device_id].local_ptr()),
      &storage.server_control, omp_operate{coarse_grain, print_buffer.get()},
      omp_clear{}));
  storage.threads.push_back(hostrpc::make_thread(&storage.thread_state.back()));

  return as_ulong(device_id);
}

// These parts are in library code as they aren't freestanding safe
// However, want to minimise the cmake needed to jury rig this into prod
#include "allocator_host_libc.cpp"
#include "allocator_hsa.cpp"
#include "hostrpc_thread.cpp"
#include "incprintf.cpp"
#endif
