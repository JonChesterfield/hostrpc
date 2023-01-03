#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>


#include "../hsa.hpp"
#include "../raiifile.hpp"
#include "../launch.hpp"

#include "crt.hpp"
#include <string.h>


demo_server server;

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

static const char *const kernel_entry = "__start.kd";

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

}  // namespace


static void callbackQueue(hsa_status_t status, hsa_queue_t *source, void *)
{
  if (status != HSA_STATUS_SUCCESS)
    {
      // may only be called with status other than success
      const char *msg = "UNKNOWN ERROR";
      hsa_status_string(status, &msg);
      fprintf(stderr, "Queue at %p inactivated due to async error:\n\t%s\n",
              source, msg);

      abort();
    }
}

static int main_with_hsa(int argc, char **argv)
{
  const bool verbose = false;
  if (argc < 2)
    {
      fprintf(stderr, "Require at least one argument\n");
      return 1;
    }

  hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();

  raiifile file(argv[1]);
  if (!file.mmapped_bytes)
    {
      fprintf(stderr, "Failed to open file %s\n", argv[1]);
      return 1;
    }

  hsa::executable ex(kernel_agent, file.mmapped_bytes, file.mmapped_length);
  if (!ex.valid())
    {
      fprintf(stderr, "HSA failed to load contents of %s\n", argv[1]);
      return 1;
    }

  printf("Loaded executable\n");
  
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



  auto gpu_locks = hsa::allocate(coarse_grained_region, slots_bytes);

  void *host_locks = aligned_alloc(64, slots_bytes); // todo: stop leaking this

  auto client_inbox = hsa::allocate(fine_grained_region, slots_bytes);
  auto client_outbox = hsa::allocate(fine_grained_region, slots_bytes);
  auto shared_buffer = hsa::allocate(fine_grained_region, slots * sizeof(BufferElement));
    


  if (!gpu_locks ||
      !client_inbox ||
      !client_outbox ||
      !shared_buffer)
    {
        fprintf(stderr, "Failed to allocate rpc buffer\n");
        exit(1);

    }

  // they're probably zero anyway. todo: check the hsa docs for the gpu one, it's more annoying to zero
  memset(host_locks, 0, slots_bytes);
  memset(client_inbox.get(), 0, slots_bytes);
  memset(client_outbox.get(), 0, slots_bytes);
  memset(shared_buffer.get(), 0, slots_bytes * sizeof(BufferElement));


  server = demo_server(
      {},
      hostrpc::careful_cast_to_bitmap<demo_server::lock_t>(host_locks,
                                                           slots_words),
      hostrpc::careful_cast_to_bitmap<demo_server::inbox_t>(client_outbox.get(),
                                                            slots_words),
      hostrpc::careful_cast_to_bitmap<demo_server::outbox_t>(client_inbox.get(),
                                                             slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer.get(), slots));

  
  // arguments must be in kernarg memory, which is constant
  // opencl doesn't accept char** as a type and returns void
  // combined, choosing to pass arguments as:
  // void* rpc_buffer[4]
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
  size_t bytes_for_kernarg = rpc_buffer_kernarg_size + 24 + implicit_offset_size + extra_implicit_size;

  auto offsets = offsets_into_strtab(app_argc, app_argv);
  size_t bytes_for_argv = 8 * app_argc;
  size_t bytes_for_strtab = (offsets.back() + 3) & ~size_t{3};
  size_t number_return_values =
      hsa::agent_get_info_wavefront_size(kernel_agent);
  size_t bytes_for_return = sizeof(int) * number_return_values;

  // Always allocates > 0 because of the return slot
  auto mutable_alloc =
      hsa::allocate(fine_grained_region,
                    bytes_for_argv + bytes_for_strtab + bytes_for_return);

  size_t strtab_start_offset = bytes_for_argv;
  
  const char *strtab_start =
      static_cast<char *>(mutable_alloc.get()) + strtab_start_offset;
  const char *result_location = static_cast<char *>(mutable_alloc.get()) +
                                strtab_start_offset + bytes_for_strtab;

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

    void *rpc_pointers[4] = {
                             gpu_locks.get(),
                             client_inbox.get(),
                             client_outbox.get(),
                             shared_buffer.get(),
    };
    memcpy(kernarg, &rpc_pointers, sizeof(rpc_pointers));
    
    
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

  if (verbose)
    {
      fprintf(stderr, "Spawn queue\n");
    }

  hsa_queue_t *queue = hsa::create_queue(kernel_agent, callbackQueue);
  if (!queue)
    {
      fprintf(stderr, "Failed to create queue\n");
      exit(1);
    }


  // Claim a packet
  uint64_t packet_id = hsa::acquire_available_packet_id(queue);

  const uint32_t mask = queue->size - 1;  // %
  hsa_kernel_dispatch_packet_t *packet =
      (hsa_kernel_dispatch_packet_t *)queue->base_address + (packet_id & mask);

  static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernarg_address) == 40, "");
  
  uint32_t wavefront_size = hsa::agent_get_info_wavefront_size(kernel_agent);
  hsa::initialize_packet_defaults(wavefront_size, packet);

  packet->workgroup_size_x = 1;
  packet->grid_size_x = 1;
   
  uint64_t kernel_address = find_entry_address(ex);
  packet->kernel_object = kernel_address;

  {
    void *raw_kernarg_alloc = kernarg_alloc.get();
    memcpy(&packet->kernarg_address, &raw_kernarg_alloc, 8);
  }

  auto rc = hsa_signal_create(1, 0, NULL, &packet->completion_signal);
  if (rc != HSA_STATUS_SUCCESS)
    {
      fprintf(stderr, "Can't make signal\n");
      exit(1);
    }

  auto m = ex.get_kernel_info();

  // todo: use hsa::launch?
  auto it = m.find(std::string(kernel_entry));
  if (it != m.end())
    {
      packet->private_segment_size = it->second.private_segment_fixed_size;
      packet->group_segment_size = it->second.group_segment_fixed_size;
    }
  else
    {
      fprintf(stderr, "Error: get_kernel_info failed for kernel %s\n",
              kernel_entry);
      exit(1);
    }

  printf("Going to try to launch a kernel\n");
  hsa::packet_store_release((uint32_t *)packet,
                            hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                            hsa::kernel_dispatch_setup());

  hsa_signal_store_release(queue->doorbell_signal, packet_id);

  if (verbose)
    {
      fprintf(stderr, "Launch kernel\n");
    }
  do
    {
      // TODO: Polling is better than waiting here as it lets the initial
      // dispatch spawn a graph

      bool r = server.rpc_handle(
          [&](hostrpc::port_t, BufferElement *data) {
            fprintf(stderr, "Server got work to do:\n");
            for (unsigned i = 0; i < 64; i++)
              {
                auto ith = data->cacheline[i];
                fprintf(stderr, "data[%u] = {%lu, %lu...}\n", i, ith.element[0],
                        ith.element[1]);
              }
          },
          [&](hostrpc::port_t, BufferElement *data) {
            fprintf(stderr, "Server cleaning up\n");
            for (unsigned i = 0; i < 64; i++)
              {
                data->cacheline[i].element[0] = 0;
              }
          });

      if (r)
        {
          // did something
          printf("server did something\n");
        }
      else
        {
          // found no work, could sleep here
          printf("no work found\n");
          for (unsigned i = 0; i < 10000; i++) platform::sleep_briefly();                      
        }
    }
  while (hsa_signal_wait_acquire(packet->completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, 5000 /*000000*/,
                                 HSA_WAIT_STATE_ACTIVE) != 0);

  if (verbose)
    {
      fprintf(stderr, "Kernel signalled\n");
    }
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

      fprintf(stderr, "Queue in x64: %lx\n", (uint64_t)queue);
      uint64_t v = ((uint64_t)result[0] & 0x00000000FFFFFFFFull) |
                   (((uint64_t)result[1] & 0x00000000FFFFFFFFull) << 32u);
      fprintf(stderr, "Queue: %lx\n", v);
      for (size_t i = 0; i < number_return_values; i++)
        {
          fprintf(stderr, "rc[%zu] = %x\n", i, result[i]);
        }
    }

  if (verbose)
    {
      fprintf(stderr, "Result[0] %d\n", result[0]);
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
