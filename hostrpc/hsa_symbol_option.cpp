// Observation: A template which is not instantiated generates no code
// Declared symbols that are not used generate no code
// If calls to cuda / hsa / libc etc are within a template declared in a header,
// and the application does not instantiate the corresponding template, it doesn't
// need to link against hsa/cuda etc
// Can therefore move the allocator specific stuff into a header file.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


#include <vector>

#include "hsa.h"
#include "hsa_ext_amd.h"

enum platform
{
  HSA,
  LIBC
};

extern "C"
{
  // doesn't include stdlib, proxy for declaring hsa functions inline
  void* malloc(size_t);
  void free(void*);
}


namespace
{
template <platform P>
struct allocator;

template <>
struct allocator<LIBC>
{
  allocator() {}

  
  void *allocate(size_t bytes) { return malloc(bytes); }
  void deallocate(void *d) { free(d); }
};

template <>
struct allocator<HSA>
{
  allocator()
  {
    hsa_pool::fine();  // force a use
  }

  void *allocate(size_t bytes)
  {
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

  void deallocate(void *d) { hsa_memory_free(d); }

 private:
  struct hsa_pool
  {
    static hsa_amd_memory_pool_t fine()
    {
      return hsa_pool::getInstance().value;
    }
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

    static std::vector<hsa_agent_t> find_agents()
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

    static hsa_amd_memory_pool_t find_fine_grain_pool_or_abort()
    {
      hsa_amd_memory_pool_t pool;
      hsa_status_t err = hsa_iterate_agents(
          [](hsa_agent_t agent, void *data) -> hsa_status_t {
            hsa_status_t err = HSA_STATUS_SUCCESS;

            hsa_device_type_t device_type;
            err =
                hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
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
                [](hsa_amd_memory_pool_t memory_pool,
                   void *data) -> hsa_status_t {
                  hsa_amd_memory_pool_t *out = (hsa_amd_memory_pool_t *)data;

                  hsa_status_t err = HSA_STATUS_SUCCESS;

                  {
                    hsa_amd_segment_t segment;
                    err = hsa_amd_memory_pool_get_info(
                        memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                        &segment);
                    if ((err != HSA_STATUS_SUCCESS) ||
                        (segment != HSA_AMD_SEGMENT_GLOBAL))
                      {
                        return err;
                      }
                  }

                  {
                    bool alloc_allowed = false;
                    err = hsa_amd_memory_pool_get_info(
                        memory_pool,
                        HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
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

                  if (0 == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED &
                            global_flag))
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
                      memory_pool,
                      HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT,
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
          //exit(1);
        }

      fprintf(stderr, "find fine_grain -> %lu\n", pool.handle);

      return pool;
    }
  };
};

}  // namespace

int a_symbol;

void user()
{
#if 1
  allocator<LIBC> alloc;
#else
  allocator<HSA> alloc;
#endif
  alloc.deallocate(alloc.allocate(12));
}
