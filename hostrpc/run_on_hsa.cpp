#include "run_on_hsa.hpp"
#include "hsa.h"
#include "hsa.hpp"
#include <assert.h>
#include <stdio.h>
#include <vector>

namespace hostrpc
{
uint32_t run_on_hsa_errcount(hsa_executable_t ex, hsa_agent_t agent,
                             hsa_queue_t *queue, void *kernarg,
                             const char *name)
{
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);

  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address +
      (packet_id & (queue->size - 1));

  hsa::initialize_packet_defaults(packet);

  hsa_executable_symbol_t symbol;
  hsa_status_t rc =
      hsa_executable_get_symbol_by_name(ex, name, &agent, &symbol);
  if (rc != HSA_STATUS_SUCCESS)
    {
      bang();
      return 1;
    }

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);

  if (kind != HSA_SYMBOL_KIND_KERNEL)
    {
      return 1;
    }

  uint64_t symbol_handle_for_dispatch =
      hsa::symbol_get_info_kernel_object(symbol);

  fprintf(stderr, "agent %lu, symbol at %lu\n", agent.handle,
          symbol_handle_for_dispatch);

  (void)agent;
  (void)queue;
  (void)kernarg;
  (void)name;
  return 0;
}

uint32_t run_on_hsa_errcount(hsa_executable_t ex, hsa_agent_t agent, void *arg,
                             size_t len, const char *name)
{
  hsa_queue_t *queue;
  {
    uint32_t min_size = hsa::agent_get_info_queue_min_size(agent);
    fprintf(stderr, "Min_size %u\n", min_size);

    hsa_status_t rc = hsa_queue_create(
        agent /* make the queue on this agent */, min_size,
        HSA_QUEUE_TYPE_MULTI, NULL /* called on every async event? */,
        NULL /* data passed to previous */,
        // If sizes exceed these values, things are supposed to work slowly
        UINT32_MAX /* private_segment_size, 32_MAX is unknown */,
        UINT32_MAX /* group segment size, as above */, &queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        bang();
        return 1;
      }
  }

  hsa_region_t kernarg_region = hsa::region_kernarg(agent);
  hsa_region_t fine_grained_region = hsa::region_fine_grained(agent);
  {
    uint64_t failure = reinterpret_cast<uint64_t>(nullptr);
    if (kernarg_region.handle == failure ||
        fine_grained_region.handle == failure)
      {
        return 1;
      }
  }

  auto fine_block = hsa::allocate(fine_grained_region, len);
  auto kernarg_block = hsa::allocate(kernarg_region, 8);
  if (!fine_block || !kernarg_block)
    {
      return 1;
    }

  // Copy arg into fine_block
  memcpy(fine_block.get(), arg, len);

  // Copy pointer to fine block into kernarg
  {
    void *f = fine_block.get();
    memcpy(kernarg_block.get(), &f, 8);
  }

  uint32_t res =
      run_on_hsa_errcount(ex, agent, queue, kernarg_block.get(), name);

  // Copy result back
  memcpy(arg, fine_block.get(), len);

  hsa_status_t rc = hsa_queue_destroy(queue);
  if (rc != HSA_STATUS_SUCCESS)
    {
      bang();
      return res + 1;
    }

  return res;
}

uint32_t run_on_hsa_errcount(hsa_executable_t ex, void *arg, size_t len,
                             const char *name)
{
  hsa::init state;
  (void)arg;
  fprintf(stderr, "Call run_on_hsa, %zu bytes, %s\n", len, name);

  std::vector<hsa_agent_t> gpus;
  hsa_status_t r = hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
    auto features = hsa::agent_get_info_feature(agent);
    if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
      {
        gpus.push_back(agent);
      }
    return HSA_STATUS_SUCCESS;
  });
  (void)r;
  fprintf(stderr, "Found %zu gpus\n", gpus.size());

  uint32_t rc = 0;
  for (size_t i = 0; i < gpus.size(); i++)
    {
      rc += run_on_hsa_errcount(ex, gpus[i], arg, len, name);
    }

  return rc;
}

void run_on_hsa(hsa_executable_t ex, void *arg, size_t len, const char *name)
{
  run_on_hsa_errcount(ex, arg, len, name);
}
}  // namespace hostrpc
