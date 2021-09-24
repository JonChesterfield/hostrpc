#include "allocator.hpp"

#include "hsa.h"
#include "hsa_ext_amd.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "detail/platform/detect.hpp"
#if !HOSTRPC_HOST
#error "allocator_hsa relies on the hsa host library"
#endif

namespace hostrpc
{
namespace allocator
{
struct hsa_pool
{
  static hsa_amd_memory_pool_t fine() { return hsa_pool::getInstance().value; }
  static hsa_status_t enable(const void *ptr)
  {
    auto &a = hsa_pool::getInstance().agents;
    fprintf(stderr, "Enable %p on %zu agents\n", ptr, a.size());

    return hsa_amd_agents_allow_access(a.size(), a.data(), nullptr, ptr);
  }

 private:
  hsa_amd_memory_pool_t value;
  std::vector<hsa_agent_t> agents;
  static hsa_pool &getInstance()
  {
    static hsa_pool instance;
    return instance;
  }
  hsa_pool()
  {
    value = find_fine_grain_pool_or_abort();
    agents = find_agents();
  }
  hsa_pool(hsa_pool const &) = delete;
  void operator=(hsa_pool const &) = delete;

  static hsa_amd_memory_pool_t find_fine_grain_pool_or_abort();
  static std::vector<hsa_agent_t> find_agents();
};

namespace hsa_impl
{
HOSTRPC_ANNOTATE int memsetzero_gpu(void *memory, size_t bytes)
{
  assert((bytes & ~(size_t)0x3) == bytes);
  hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
  if (r != HSA_STATUS_SUCCESS)
    {
      return 1;
    }
  return 0;
}

void *allocate(uint64_t hsa_region_t_handle, size_t align, size_t bytes)
{
  assert(align >= 64);
  assert(align <= 4096);
  (void)align;
  hsa_region_t region{.handle = hsa_region_t_handle};

  bytes = 4 * ((bytes + 3) / 4);  // fill uses a multiple of four

  void *memory;
  if (HSA_STATUS_SUCCESS == hsa_memory_allocate(region, bytes, &memory))
    {
      // probably want memset for fine grain, may want it for gfx9
      // memset(memory, 0, bytes);
      // warning: This is likely to be relied on by bitmap
      hsa_status_t r = hsa_amd_memory_fill(memory, 0, bytes / 4);
      if (HSA_STATUS_SUCCESS == r)
        {
          return memory;
        }
    }

  return nullptr;
}

void *allocate_fine_grain(size_t bytes)
{
  bytes = 4 * ((bytes + 3) / 4);  // fill uses a multiple of four

  hsa_amd_memory_pool_t pool = hsa_pool::fine();

  void *memory;
  hsa_status_t rc = hsa_amd_memory_pool_allocate(pool, bytes, 0, &memory);

  if (rc != HSA_STATUS_SUCCESS)
    {
      printf("allocate_fine_grain: Fail at line %u\n", __LINE__);
      return nullptr;
    }

  rc = hsa_pool::enable(memory);
  if (rc != HSA_STATUS_SUCCESS)
    {
      printf("allocate_fine_grain: Fail at line %u\n", __LINE__);
      return nullptr;
    }

  return memory;
}

int deallocate(void *d)
{
  return (hsa_memory_free(d) == HSA_STATUS_SUCCESS) ? 0 : 1;
}

}  // namespace hsa_impl

std::vector<hsa_agent_t> hsa_pool::find_agents()
{
  std::vector<hsa_agent_t> res;
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        std::vector<hsa_agent_t> *res = (std::vector<hsa_agent_t> *)data;
        res->push_back(agent);
        return HSA_STATUS_SUCCESS;
      },
      &res);
  return res;
}

hsa_amd_memory_pool_t hsa_pool::find_fine_grain_pool_or_abort()
{
  hsa_amd_memory_pool_t pool;
  hsa_status_t err = hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) -> hsa_status_t {
        hsa_status_t err = HSA_STATUS_SUCCESS;

        hsa_device_type_t device_type;
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if (err != HSA_STATUS_SUCCESS)
          {
            return err;
          }

        if (device_type != HSA_DEVICE_TYPE_CPU)
          {
            return HSA_STATUS_SUCCESS;
          }

        return hsa_amd_agent_iterate_memory_pools(
            agent,
            [](hsa_amd_memory_pool_t memory_pool, void *data) -> hsa_status_t {
              hsa_amd_memory_pool_t *out = (hsa_amd_memory_pool_t *)data;

              hsa_status_t err = HSA_STATUS_SUCCESS;

              {
                hsa_amd_segment_t segment;
                err = hsa_amd_memory_pool_get_info(
                    memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                if ((err != HSA_STATUS_SUCCESS) ||
                    (segment != HSA_AMD_SEGMENT_GLOBAL))
                  {
                    return err;
                  }
              }

              {
                bool alloc_allowed = false;
                err = hsa_amd_memory_pool_get_info(
                    memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                    &alloc_allowed);
                if ((err != HSA_STATUS_SUCCESS) || !alloc_allowed)
                  {
                    return err;
                  }
              }

              {
                size_t size;
                err = hsa_amd_memory_pool_get_info(
                    memory_pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
                if ((err != HSA_STATUS_SUCCESS) || (size == 0))
                  {
                    return err;
                  }
              }

              uint32_t global_flag = 0;
              err = hsa_amd_memory_pool_get_info(
                  memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                  &global_flag);
              if (err != HSA_STATUS_SUCCESS)
                {
                  return err;
                }

              if (0 ==
                  (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED & global_flag))
                {
                  return HSA_STATUS_SUCCESS;
                }

              {
                bool accessible;
                err = hsa_amd_memory_pool_get_info(
                    memory_pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL,
                    &accessible);
                if ((err != HSA_STATUS_SUCCESS) || !accessible)
                  {
                    return err;
                  }
              }

              // Located a memory pool meeting various requirements
              size_t alignment;
              err = hsa_amd_memory_pool_get_info(
                  memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
                  &alignment);

              if (alignment < 4096)
                {
                  return HSA_STATUS_ERROR;
                }

              *out = memory_pool;

              return HSA_STATUS_INFO_BREAK;
            },
            data);
      },
      &pool);

  if (err != HSA_STATUS_INFO_BREAK)
    {
      fprintf(stderr, "allocator_hsa failed first init\n");
      exit(1);
    }

  fprintf(stderr, "find fine_grain -> %lu\n", pool.handle);

  return pool;
}

}  // namespace allocator

void init() { allocator::hsa_pool::fine(); }
}  // namespace hostrpc
