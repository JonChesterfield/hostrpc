

#include "detail/common.hpp"
#include "detail/platform.hpp"
#include "detail/platform_detect.hpp"
#include "hostcall.hpp"
#include "hostcall_hsa.hpp"
#include "x64_gcn_type.hpp"

#include "detail/client_impl.hpp"
#include "detail/server_impl.hpp"

enum opcodes
{
  opcodes_nop = 0,
  opcodes_malloc = 1,
  opcodes_free = 2,
};

#if HOSTRPC_AMDGCN

// overrides weak functions in target_impl.hip
extern "C"
{
  void *__kmpc_impl_malloc(size_t);
  void __kmpc_impl_free(void *);
}

using client_type = hostrpc::x64_gcn_type<hostrpc::size_runtime>::client_type;

client_type *get_client()
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
  // it's written into a point in the implicit arguments where the GPU can
  // retrieve it from
  // size_t* argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  // result is found in argptr[3]

  size_t *argptr = (size_t *)__builtin_amdgcn_implicitarg_ptr();
  return (client_type *)argptr[3];
}

struct fill
{
  uint64_t *d;
  fill(uint64_t *d) : d(d) {}
  void operator()(hostrpc::page_t *page)
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
  void operator()(hostrpc::page_t *page)
  {
    hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
    for (unsigned i = 0; i < 8; i++)
      {
        d[i] = line->element[i];
      }
  }
};

extern "C"
{
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
        success = c->rpc_invoke(f);
      }
  }
}

#endif

#if HOSTRPC_HOST

#include "hsa.hpp"
#include <list>

// overrides weak functions in rtl.cpp
extern "C"
{
  // gets called repeatedly
  // uses agent to size queue
  unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                      uint32_t device_id);
  hsa_status_t hostrpc_init();
  hsa_status_t hostrpc_terminate();
}

struct operate
{
  static void op(hostrpc::cacheline_t *line)
  {
    uint64_t op = line->element[0];
    switch (op)
      {
        case opcodes_nop:
          {
            break;
          }
        case opcodes_malloc:
          {
            uint64_t size;
            __builtin_memcpy(&size, &line->element[1], 8);
            // needs a memory region derived from a kernel_agent
            (void)size;  // todo
            break;
          }
        case opcodes_free:
          {
            void *ptr;
            __builtin_memcpy(&ptr, &line->element[1], 8);
            hsa_memory_free(ptr);
            break;
          }
      }
    return;
  }

  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        hostrpc::cacheline_t *line = &page->cacheline[c];
        op(line);
      }
  }
};

struct clear
{
  void operator()(hostrpc::page_t *page)
  {
    for (unsigned c = 0; c < 64; c++)
      {
        page->cacheline[c].element[0] = opcodes_nop;
      }
  }
};

struct storage_t
{
  using type = hostrpc::x64_gcn_type<hostrpc::size_runtime>;
  std::vector<type> stash;
  std::vector<type::storage_type::AllocLocal::raw> server_pointers;
  std::vector<type::storage_type::AllocRemote::raw> client_pointers;

  storage_t() = default;
  ~storage_t()
  {
    size_t N = server_pointers.size();
    assert(N == client_pointers.size());
    for (size_t i = 0; i < N; i++)
      {
        // todo: ensure destroy can be called on non-valid pointers
        server_pointers[i].destroy();
        client_pointers[i].destroy();
      }
  }
} storage;

unsigned long hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                                    uint32_t device_id)
{
  (void)this_Q;
  (void)device_id;

  while (storage.client_pointers.size() < device_id)
    {
      storage.stash.push_back({});
      storage.server_pointers.push_back({});
      storage.client_pointers.push_back({});
    }

  auto as_ulong = [](uint32_t device_id) -> unsigned long {
    // assumes null is zero, todo: can that be dropped
    void *p = storage.client_pointers[device_id].remote_ptr();
    unsigned long t;
    __builtin_memcpy(&t, &p, 8);
    return t;
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

  auto size = hostrpc::size_runtime(numCu * waverPerCu);
  hsa_region_t fine_grain = hsa::region_fine_grained(agent);
  hsa_region_t coarse_grain = hsa::region_coarse_grained(agent);

  storage.stash[device_id] =
      storage_t::type{size, fine_grain.handle, coarse_grain.handle};

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
    auto alloc =
        storage_t::type::storage_type::AllocRemote(coarse_grain.handle);
    storage.client_pointers[device_id] =
        alloc.allocate(sizeof(storage_t::type::client_type));

    auto remote = storage.client_pointers[device_id].remote_ptr();
    int cp =
        hsa::copy_host_to_gpu(agent, remote, &storage.stash[device_id].client,
                              sizeof(storage_t::type::client_type));
    if (!cp)
      {
        // bad, need to do some cleanup

        return 0;
      }
  }

  // finally need to spawn a thread to run the server process

  return as_ulong(device_id);
}

hsa_status_t hostrpc_init() { return HSA_STATUS_SUCCESS; }
hsa_status_t hostrpc_terminate()
{
  // storage = {}; // wants various copy constructors defined on storage, drop it for now
  return HSA_STATUS_SUCCESS;
}

#endif


#if HOSTRPC_HOST
int main()
{

}
#endif
