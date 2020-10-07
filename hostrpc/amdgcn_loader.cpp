#include "hsa.hpp"
#include <cassert>
#include <cstring>
#include <vector>

#include <unistd.h>

#include "hostcall.hpp"

namespace
{
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

static const char *const kernel_entry = "__device_start.kd";

uint64_t find_entry_address(hsa::executable &ex)
{
  hsa_executable_symbol_t symbol = ex.get_symbol_by_name(kernel_entry);
  if (symbol.handle == hsa::sentinel())
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
}  // namespace

static int main_with_hsa(int argc, char **argv)
{
  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
      return 1;
    }

  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  // Load argv[1]
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

  // probably need to populate some of the implicit args for intrinsics to work
  hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
  hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
  hsa_region_t coarse_grained_region = hsa::region_coarse_grained(kernel_agent);
  {
    uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
    if (kernarg_region.handle == fail || fine_grained_region.handle == fail ||
        coarse_grained_region.handle == fail)
      {
        fprintf(stderr, "Failed to find allocation region on kernel agent\n");
        exit(1);
      }
  }

  // Drop the loader name from the forwarded argc/argv
  int app_argc = argc - 1;
  char **app_argv = &argv[1];

  // arguments must be in kernarg memory, which is constant
  // opencl doesn't accept char** as a type and returns void
  // combined, choosing to pass arguments as:
  // int argc
  // int padding
  // void * to_argv
  // int * to_result

  // there's also a number of implicit arguments, where those passed by atmi
  // don't match those I see from an opencl kernel. The first 24 bytes are
  // consistently used for offset_x, offset_y, offset_z. Zero those.
  // opencl and atmi both think the implicit structure is 80 long.

  size_t implicit_offset_size = 24;
  size_t extra_implicit_size = 80 - implicit_offset_size;

  // implicit offset needs to be 8 byte aligned, which it w/ 24 bytes explicit
  size_t bytes_for_kernarg = 24 + implicit_offset_size + extra_implicit_size;

  auto offsets = offsets_into_strtab(app_argc, app_argv);
  size_t bytes_for_argv = 8 * app_argc;
  size_t bytes_for_strtab = (offsets.back() + 3) & ~size_t{3};
  size_t number_return_values = 64;  // max number waves
  size_t bytes_for_return = sizeof(int) * number_return_values;

  // Always allocates > 0 because of the return slot
  auto mutable_alloc =
      hsa::allocate(fine_grained_region,
                    bytes_for_argv + bytes_for_strtab + bytes_for_return);

  const char *strtab_start =
      static_cast<char *>(mutable_alloc.get()) + bytes_for_argv;
  const char *result_location = static_cast<char *>(mutable_alloc.get()) +
                                bytes_for_argv + bytes_for_strtab;

  auto kernarg_alloc = hsa::allocate(kernarg_region, bytes_for_kernarg);
  if (!mutable_alloc || !kernarg_alloc)
    {
      fprintf(stderr, "Failed to allocate %zu bytes for kernel arguments\n",
              bytes_for_argv + bytes_for_strtab + bytes_for_kernarg);
      exit(1);
    }

  // Populate argv array, immediately followed by string table
  char *argv_array = static_cast<char *>(mutable_alloc.get());
  for (int i = 0; i < app_argc; i++)
    {
      const char *loc = strtab_start + offsets[i];
      memcpy(argv_array, &loc, 8);
      argv_array += 8;
    }
  for (int i = 0; i < app_argc; i++)
    {
      char *arg = app_argv[i];
      size_t sz = strlen(arg) + 1;
      memcpy(argv_array, arg, sz);
      argv_array += sz;
    }

  for (unsigned i = 0; i < bytes_for_strtab - offsets.back(); i++)
    {
      // alignment padding for the return value
      char z = 0;
      memcpy(argv_array, &z, 1);
      argv_array += 1;
    }

  // init the return value. not strictly necessary
  {
    assert(argv_array == result_location);
    for (size_t i = 0; i < number_return_values; i++)
      {
        unsigned z = 0xdead;
        memcpy(argv_array, &z, 4);
        argv_array += 4;
      }
  }

  // Set up kernel arguments
  {
    char *kernarg = (char *)kernarg_alloc.get();

    // argc
    memcpy(kernarg, &app_argc, 4);
    kernarg += 4;

    // padding
    memset(kernarg, 0, 4);
    kernarg += 4;

    // argv
    void *raw_mutable_alloc = mutable_alloc.get();
    memcpy(kernarg, &raw_mutable_alloc, 8);
    kernarg += 8;

    // result
    memcpy(kernarg, &result_location, 8);
    kernarg += 8;

    // x, y, z implicit offsets
    memset(kernarg, 0, implicit_offset_size);
    kernarg += implicit_offset_size;

    // remaining implicit gets scream. I don't think the kernels are
    // using it, but if they do, -1 is relatively obvious in the dump
    memset(kernarg, 0xff, extra_implicit_size);
    kernarg += extra_implicit_size;
  }

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
        exit(1);
      }
  }

  hostcall hc(ex, kernel_agent);
  if (!hc.valid())
    {
      fprintf(stderr, "Failed to create hostcall\n");
      exit(1);
    }
  if (hc.enable_queue(queue) != 0)
    {
      fprintf(stderr, "Failed to enable queue\n");
      exit(1);
    }
  for (unsigned r = 0; r < 2; r++)
    {
      if (hc.spawn_worker(queue) != 0)
        {
          fprintf(stderr, "Failed to spawn worker\n");
          exit(1);
        }
    }

  // Claim a packet
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);

  const uint32_t mask = queue->size - 1;  // %
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address + (packet_id & mask);

  hsa::initialize_packet_defaults(packet);

  uint64_t kernel_address = find_entry_address(ex);
  packet->kernel_object = kernel_address;

  {
    void *raw_kernarg_alloc = kernarg_alloc.get();
    memcpy(&packet->kernarg_address, &raw_kernarg_alloc, 8);
  }

  auto rc = hsa_signal_create(1, 0, NULL, &packet->completion_signal);
  if (rc != HSA_STATUS_SUCCESS)
    {
      printf("Can't make signal\n");
      exit(1);
    }

  auto m = ex.get_kernel_info();

  auto it = m.find(std::string(kernel_entry));
  if (it != m.end())
    {
      packet->private_segment_size = it->second.private_segment_fixed_size;
      packet->group_segment_size = it->second.group_segment_fixed_size;
    }
  else
    {
      printf("Error: get_kernel_info failed\n");
      exit(1);
    }

  packet_store_release((uint32_t *)packet,
                       header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                       kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  do
    {
      // TODO: Polling is better than waiting here as it lets the initial
      // dispatch spawn a graph
    }
  while (hsa_signal_wait_acquire(packet->completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, 5000 /*000000*/,
                                 HSA_WAIT_STATE_ACTIVE) != 0);

  int result[number_return_values];
  memcpy(&result, result_location, sizeof(int) * number_return_values);

  hsa_signal_destroy(packet->completion_signal);
  hsa_queue_destroy(queue);

  bool results_match = true;
  {
    int res = result[0];
    for (size_t i = 1; i < number_return_values; i++)
      {
        if (result[i] != res)
          {
            results_match = false;
          }
      }
  }

  if (!results_match)
    {
      fprintf(stderr, "Warning: Non-uniform return values\n");

      printf("Queue in x64: %lx\n", (uint64_t)queue);
      uint64_t v = ((uint64_t)result[0] & 0x00000000FFFFFFFFull) |
                   (((uint64_t)result[1] & 0x00000000FFFFFFFFull) << 32u);
      printf("Queue: %lx\n", v);
      for (size_t i = 0; i < number_return_values; i++)
        {
          fprintf(stderr, "rc[%zu] = %x\n", i, result[i]);
        }
    }

  return result[0];
}

extern "C" int amdgcn_loader_main(int argc, char **argv)
{
  // valgrind thinks this is leaking slightly
  hsa::init hsa_state;  // Need to destroy this last
  return main_with_hsa(argc, argv);
}

__attribute__((weak)) extern "C" int main(int argc, char **argv)
{
  return amdgcn_loader_main(argc, argv);
}
