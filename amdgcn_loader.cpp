#include "hsa.hpp"
#include <cstring>

uint64_t find_entry_address(hsa::executable &ex)
{
  const char *kernel_entry = "device_entry.kd";
  hsa_executable_symbol_t symbol = ex.get_symbol_by_name(kernel_entry);
  if (symbol.handle == hsa::executable::sentinel())
    {
      fprintf(stderr, "HSA failed to find kernel %s\n", kernel_entry);
      exit(1);
    }

  hsa_symbol_kind_t kind = hsa::symbol_get_info_type(symbol);
  if (kind != HSA_SYMBOL_KIND_KERNEL)
    {
      fprintf(stderr, "Symbol %s is not a kernel\n", kernel_entry);
      exit(1);
    }

  return hsa::symbol_get_info_kernel_object(symbol);
}

void initialize_packet_defaults(hsa_kernel_dispatch_packet_t *packet)
{
  // Reserved fields, private and group memory, and completion signal are all
  // set to 0.
  memset(((uint8_t *)packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);
  // These values should probably be read from the kernel
  // Currently they're copied from documentation
  packet->workgroup_size_x = 256;
  packet->workgroup_size_y = 1;
  packet->workgroup_size_z = 1;
  packet->grid_size_x = 256;
  packet->grid_size_y = 1;
  packet->grid_size_z = 1;

  // These definitely get overwritten by the caller
  packet->kernel_object = 0;  //  KERNEL_OBJECT;
  packet->kernarg_address = NULL;
}

void packet_store_release(uint32_t *packet, uint16_t header, uint16_t rest)
{
  __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
}

uint16_t header(hsa_packet_type_t type)
{
  uint16_t header = type << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  return header;
}

uint16_t kernel_dispatch_setup()
{
  return 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
}

int main(int argc, char **argv)
{
  hsa::init hsa_state;

  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
    }

  hsa_agent_t kernel_agent;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        auto features = hsa::agent_get_info_feature(agent);
        if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
          {
            kernel_agent = agent;
            return HSA_STATUS_INFO_BREAK;
          }
        return HSA_STATUS_SUCCESS;
      }))
    {
      fprintf(stderr, "Failed to find a kernel agent\n");
      return 1;
    }

  FILE *fh = fopen(argv[1], "rb");
  int fn = fh ? fileno(fh) : -1;
  if (fn < 0)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  hsa::executable ex(kernel_agent, fn);
  if (!ex.valid())
    {
      fprintf(stderr, "HSA failed to load contents of %s\n", argv[1]);
      return 1;
    }

  uint64_t kernel_address = find_entry_address(ex);
  printf("Kernel is at %lx\n", kernel_address);

  hsa_queue_t *queue;
  {
    hsa_status_t rc = hsa_queue_create(
        kernel_agent /* make the queue on this agent */,
        131072 /* todo: size it, this hardcodes max size for vega20 */,
        HSA_QUEUE_TYPE_SINGLE /* baseline */,
        NULL /* called on every async event? */,
        NULL /* data passed to previous */,
        // If sizes exceed these values, things are supposed to work slowly
        UINT32_MAX /* private_segment_size, 32_MAX is unknown */,
        UINT32_MAX /* group segment size, as above */, &queue);
    if (rc != HSA_STATUS_SUCCESS)
      {
        fprintf(stderr, "Failed to create queue\n");
      }
  }

  uint64_t packet_id = hsa_queue_add_write_index_relaxed(queue, 1);

  // Calculate the virtual address where to place the packet
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address + packet_id;

  // Populate fields in kernel dispatch packet, except for the header, the
  // setup, and the completion signal fields

  // need iterate regions to find where to allocate kernel arguments

  auto get_kernarg = [](hsa_region_t region, void *data) -> hsa_status_t {
    hsa_region_segment_t segment;
    hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (segment != HSA_REGION_SEGMENT_GLOBAL)
      {
        return HSA_STATUS_SUCCESS;
      }

    hsa_region_global_flag_t flags = hsa_region_get_info_flags(region);
    if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
      {
        hsa_region_t *ret = (hsa_region_t *)data;
        *ret = region;
        return HSA_STATUS_INFO_BREAK;
      }
    return HSA_STATUS_SUCCESS;
  };

  (void)get_kernarg;

  initialize_packet_defaults(packet);
  packet->kernel_object = kernel_address;

  hsa_signal_create(1, 0, NULL, &packet->completion_signal);

  printf("Launching kernel\n");

  packet_store_release((uint32_t *)packet,
                       header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  while (hsa_signal_wait_acquire(packet->completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                 HSA_WAIT_STATE_ACTIVE) != 0)
    ;

  return 0;
}
