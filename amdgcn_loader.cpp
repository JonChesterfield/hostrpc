#include "hsa.hpp"
#include <cstring>
#include <vector>

size_t bytes_for_argv_strtab(int argc, char **argv)
{
  size_t count = 0;
  for (int i = 0; i < argc; i++)
    {
      char *arg = argv[i];
      count += strlen(arg) + 1;
    }
  return count;
}

std::vector<size_t> offsets_into_strtab(int argc, char **argv)
{
  std::vector<size_t> res;
  unsigned offset = 0;
  for (int i = 0; i < argc; i++)
    {
      char *arg = argv[i];
      size_t sz = strlen(arg) + 1;
      res.push_back(offset);
      offset += sz;
    }

  // And an end element for the total size
  res.push_back(offset);
  return res;
}

void *allocate_and_populate_argc_argv(hsa_agent_t kernel_agent, int app_argc,
                                      char **app_argv)
{
  const bool verbose = false;
  hsa_region_t kernarg_region;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_regions(
          kernel_agent, [&](hsa_region_t region) -> hsa_status_t {
            hsa_region_segment_t segment = hsa::region_get_info_segment(region);
            if (segment != HSA_REGION_SEGMENT_GLOBAL)
              {
                return HSA_STATUS_SUCCESS;
              }

            hsa_region_global_flag_t flags =
                hsa::region_get_info_global_flags(region);
            if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
              {
                kernarg_region = region;
                return HSA_STATUS_INFO_BREAK;
              }
            return HSA_STATUS_SUCCESS;
          }))
    {
      fprintf(stderr, "Failed to find kernarg_region on kernel agent\n");
      exit(1);
    }

  auto offsets = offsets_into_strtab(app_argc, app_argv);

  size_t bytes_for_argc = 8;
  size_t bytes_for_argv = 8 /* the char** */ + 8 * app_argc;
  size_t bytes_for_strtab = offsets.back();
  size_t kernarg_size = bytes_for_argc + bytes_for_argv + bytes_for_strtab;

  void *kernarg_address;
  {
    if (HSA_STATUS_SUCCESS != hsa_memory_allocate(kernarg_region, kernarg_size,
                                                  (void **)&kernarg_address))
      {
        fprintf(stderr, "Failed to allocate %zu bytes for kernel arguments\n",
                kernarg_size);
        exit(1);
      }
  }

  static_assert(sizeof(int) == 4, "");

  char *kernarg = (char *)kernarg_address;
  {
    int z = 0;
    memcpy(kernarg, &app_argc, 4);
    kernarg += 4;
    memcpy(kernarg, &z, 4);
    kernarg += 4;
  }
  const char *argv_start = (char *)kernarg_address + bytes_for_argc;

  const char *strtab_start = argv_start + bytes_for_argv;

  if (verbose)
    {
      printf("kernarg %lu\n", (uint64_t)kernarg_address);
      printf("argv_start at %lu (%lu)\n", (uint64_t)argv_start,
             (uint64_t)argv_start - (uint64_t)kernarg_address);
      printf("strtab_start at %lu (%lu)\n", (uint64_t)strtab_start,
             (uint64_t)strtab_start - (uint64_t)kernarg_address);
    }

  {
    const char *argv_payload = argv_start + 8;
    memcpy(kernarg, &argv_payload, 8);
    kernarg += 8;
  }
  for (int i = 0; i < app_argc; i++)
    {
      const char *loc = strtab_start + offsets[i];
      if (verbose)
        {
          printf("Expecting %s at %lu (%lu)\n", app_argv[i], (uint64_t)loc,
                 (uint64_t)loc - (uint64_t)kernarg_address);
        }
      memcpy(kernarg, &loc, 8);
      kernarg += 8;
    }
  for (int i = 0; i < app_argc; i++)
    {
      char *arg = app_argv[i];
      size_t sz = strlen(arg) + 1;
      if (verbose)
        {
          printf("copying app_argv[%d] = %s to %lu (%lu)\n", i, arg,
                 (uint64_t)kernarg,
                 (uint64_t)kernarg - (uint64_t)kernarg_address);
        }
      memcpy(kernarg, arg, sz);
      kernarg += sz;
    }

  return kernarg_address;
}

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

  hsa_region_t kernarg_region;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_regions(
          kernel_agent, [&](hsa_region_t region) -> hsa_status_t {
            hsa_region_segment_t segment = hsa::region_get_info_segment(region);
            if (segment != HSA_REGION_SEGMENT_GLOBAL)
              {
                return HSA_STATUS_SUCCESS;
              }

            hsa_region_global_flag_t flags =
                hsa::region_get_info_global_flags(region);
            if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
              {
                kernarg_region = region;
                return HSA_STATUS_INFO_BREAK;
              }
            return HSA_STATUS_SUCCESS;
          }))
    {
      fprintf(stderr, "Failed to find kernarg_region on kernel agent\n");
      exit(1);
    }

  initialize_packet_defaults(packet);
  packet->kernel_object = kernel_address;

  // probably need to populate some of the implicit args, need to check what the
  // abi says about int followed by char**

  // Drop the loader name from the forwarded argc/argv
  {
    int app_argc = argc - 1;
    char **app_argv = &argv[1];
    void *kernarg_address =
        allocate_and_populate_argc_argv(kernel_agent, app_argc, app_argv);
    memcpy(&packet->kernarg_address, &kernarg_address, 8);

    if (false)
      {
        printf("Rendered argc/argv:\n");
        char *from = (char *)packet->kernarg_address;
        int c;
        memcpy(&c, from, 4);
        from += 4;
        // zero
        from += 4;
        char **v;
        memcpy(&v, from, 8);
        from += 8;
        printf("argc %d\n", c);
        for (int i = 0; i < c; i++)
          {
            printf("argv[%d] = %s at %lu (%lu)\n", i, v[i], (uint64_t)v[i],
                   (uint64_t)v[i] - (uint64_t)packet->kernarg_address);
          }
      }
  }

  hsa_signal_create(1, 0, NULL, &packet->completion_signal);

  printf("Launching kernel\n");

  packet_store_release((uint32_t *)packet,
                       header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  while (hsa_signal_wait_acquire(packet->completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                 HSA_WAIT_STATE_ACTIVE) != 0)
    {
      // TODO: Run a hostcall server in here
    }

  hsa_signal_destroy(packet->completion_signal);
  hsa_queue_destroy(queue);

  return 0;
}
