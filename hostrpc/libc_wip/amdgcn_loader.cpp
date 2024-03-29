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


static __libc_rpc_server server;

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

hsa_amd_memory_pool_t find_fine_grain_pool_or_exit()
{
  hsa_agent_t cpu_agent;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_agents([&](hsa_agent_t agent) -> hsa_status_t {
        auto features = hsa::agent_get_info_feature(agent);
        if (!(features & HSA_AGENT_FEATURE_KERNEL_DISPATCH))
          {
            cpu_agent = agent;
            return HSA_STATUS_INFO_BREAK;
          }
        return HSA_STATUS_SUCCESS;
      }))
    {
      fprintf(stderr, "Failed to find a cpu agent\n");
      exit(1);
    }

  hsa_amd_memory_pool_t fine_grain_pool;
  if (HSA_STATUS_INFO_BREAK !=
      hsa::iterate_memory_pools(
          cpu_agent, [&](hsa_amd_memory_pool_t pool) -> hsa_status_t {
            hsa_amd_segment_t segment;


            if (hsa_status_t rc = hsa_amd_memory_pool_get_info(
                                                  pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment))
              {
                return rc;
              }

            if (segment != HSA_AMD_SEGMENT_GLOBAL)
              {
                return HSA_STATUS_SUCCESS;
              }

            uint32_t val;
            if (hsa_status_t rc = hsa_amd_memory_pool_get_info(
                                                  pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &val))
              {
                return rc;
              }

            printf("Found a pool, flags %u\n", val);

            if (val & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
              // This one is acceptable
              fine_grain_pool = pool;
              return HSA_STATUS_INFO_BREAK;
            }
            
            return HSA_STATUS_SUCCESS;
          }))
    {
      fprintf(stderr, "Failed to find a memory pool agent\n");
      exit(1);
    }

  return fine_grain_pool;

}

struct mmapped_pair
{
  mmapped_pair(hsa_agent_t kernel_agent, size_t N, int * maybe_handle)
  {

    // Locking is optional
    // Write access is not, memory_lock fails with only PROT_READ set
    if (maybe_handle)
      {
        host = mmap(NULL, N, PROT_READ | PROT_WRITE, MAP_SHARED |  MAP_LOCKED, *maybe_handle, 0);
      }    
    else
      {
        host = mmap(NULL, N, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);

      }
    if (host == MAP_FAILED) {
      return;
    }

    hsa_amd_memory_pool_t fine_grain_pool = find_fine_grain_pool_or_exit(); // cache it?

  hsa_agent_t both[] = {kernel_agent,
                        // cpu_agent, // doesn't seem to need this
  };

    if (hsa_status_t rc = hsa_amd_memory_lock_to_pool(host, N, both, sizeof(both)/sizeof(both[0]), fine_grain_pool, 0, &gpu))
      {
        munmap(host, N);        
        host = MAP_FAILED;
        return;
      }
    
    valid = true;
  }

  ~mmapped_pair()
  {
    if (!valid) return;

    hsa_status_t rc = hsa_amd_memory_unlock(host);
    if (rc != HSA_STATUS_SUCCESS) { /* should be unreachable */ return; }

    munmap(host, N);
  }

  size_t N;
  bool valid = false;
  void * host = nullptr;
  void * gpu = nullptr;

};

struct file_handles
{
  int client_inbox;
  int shared;
  int client_outbox;
};

static int main_with_hsa(int argc, char **argv, file_handles *maybe_handles)
{
  enum {number_gpu_threads = 1};
  
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


#define USE_MMAP 1

  if (maybe_handles != nullptr)
    {
#if ! (USE_MMAP)
      fprintf(stderr, "Can't use file handles with hsa::allocate\n");
      exit(1);
#endif
    }
  
#if ! (USE_MMAP)
  auto client_inbox_store = hsa::allocate(fine_grained_region, slots_bytes);
  auto client_outbox_store = hsa::allocate(fine_grained_region, slots_bytes);

  void *client_inbox_host = client_inbox_store.get();
  void *client_inbox_gpu = client_inbox_store.get();

  void *client_outbox_host = client_outbox_store.get();
  void *client_outbox_gpu = client_outbox_store.get();
  
  auto shared_buffer_store = hsa::allocate(fine_grained_region, slots * sizeof(BufferElement));
  void *shared_buffer_host = shared_buffer_store.get();
  void *shared_buffer_gpu = shared_buffer_store.get();

#else
  mmapped_pair client_inbox_store (kernel_agent, slots_bytes, maybe_handles ? &maybe_handles->client_inbox : nullptr);
  if (!client_inbox_store.valid) {
    
        fprintf(stderr, "Failed to mmap rpc buffer\n");
        exit(1);
    }
  void *client_inbox_gpu = client_inbox_store.gpu;
  void *client_inbox_host = client_inbox_store.host;

  mmapped_pair client_outbox_store (kernel_agent, slots_bytes,  maybe_handles ? &maybe_handles->client_outbox : nullptr);
  if (!client_outbox_store.valid) {
    
        fprintf(stderr, "Failed to mmap rpc buffer\n");
        exit(1);
    }
  void *client_outbox_gpu = client_outbox_store.gpu;
  void *client_outbox_host = client_outbox_store.host;

  mmapped_pair shared_buffer_store (kernel_agent, slots * sizeof(BufferElement), maybe_handles ? &maybe_handles->shared : nullptr);
  if (!shared_buffer_store.valid) {
    
        fprintf(stderr, "Failed to mmap rpc buffer\n");
        exit(1);
    }
  void *shared_buffer_gpu = shared_buffer_store.gpu;
  void *shared_buffer_host = shared_buffer_store.host;

#endif  
  


  if (!gpu_locks ||
      !client_inbox_host ||
      !client_outbox_host ||
      !shared_buffer_host)
    {
        fprintf(stderr, "Failed to allocate rpc buffer\n");
        exit(1);

    }

  // they're probably zero anyway. todo: check the hsa docs for the gpu one, it's more annoying to zero
  memset(host_locks, 0, slots_bytes);
  memset(client_inbox_host, 0, slots_bytes);
  memset(client_outbox_host, 0, slots_bytes);
  memset(shared_buffer_host, 0, slots_bytes * sizeof(BufferElement));


  server = __libc_rpc_server(
      {},
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::lock_t>(host_locks,
                                                           slots_words),
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::inbox_t>(client_outbox_host,
                                                            slots_words),
      hostrpc::careful_cast_to_bitmap<__libc_rpc_server::outbox_t>(client_inbox_host,
                                                             slots_words),
      hostrpc::careful_array_cast<BufferElement>(shared_buffer_host, slots));
  
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
  if (number_gpu_threads < number_return_values)
    {
      number_return_values = number_gpu_threads;
    }
  
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
                             client_inbox_gpu,
                             client_outbox_gpu,
                             shared_buffer_gpu,
    };
    memcpy(kernarg, &rpc_pointers, sizeof(rpc_pointers));
    kernarg += sizeof(rpc_pointers);
    
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

  packet->workgroup_size_x = number_gpu_threads; // assuming single warp for now
  packet->grid_size_x = packet->workgroup_size_x;

  uint64_t kernel_address = find_entry_address(ex);
  packet->kernel_object = kernel_address;

  {
    void *raw_kernarg_alloc = kernarg_alloc.get();
    memcpy(&packet->kernarg_address, &raw_kernarg_alloc, 8);
  }

  hsa_signal_t completion_signal;
  auto rc = hsa_signal_create(1, 0, NULL, &completion_signal);
  packet->completion_signal = completion_signal;
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

      if (maybe_handles != nullptr) { continue; }

      bool r = //server.
        rpc_handle(&server,
          [&](uint32_t, BufferElement *data) {
            fprintf(stderr, "Server got work to do:\n");

            for (unsigned i = 0; i < 64; i++)
              {
                auto ith = data->cacheline[i];
                uint64_t opcode = ith.element[0];
                switch(opcode)
                  {
                  case no_op:
                  default:
                    continue;
                  case print_to_stderr:
                    enum {w = 7 * 8};
                    char buf[w];
                    memcpy(buf, &ith.element[1], w);
                    buf[w-1] = '\0';
                    fprintf(stderr, "%s", buf);
                    break;
                  }
              }

            fprintf(stderr, "\n");
          },
          [&](uint32_t, BufferElement *data) {
            fprintf(stderr, "Server cleaning up\n");
            for (unsigned i = 0; i < 64; i++)
              {
                data->cacheline[i].element[0] = no_op;
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
  while (hsa_signal_wait_acquire(completion_signal,
                                 HSA_SIGNAL_CONDITION_EQ, 0, 5000 /*000000*/,
                                 HSA_WAIT_STATE_ACTIVE) != 0);

  if (verbose)
    {
      fprintf(stderr, "Kernel signalled\n");
    }
  int result[number_return_values];
  memcpy(&result, result_location, sizeof(int) * number_return_values);

  hsa_signal_destroy(completion_signal);
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
          fprintf(stderr, "rc[%zu] = %d\n", i, result[i]);
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
  return main_with_hsa(argc, argv, nullptr);
}

__attribute__((weak)) extern "C" int main(int argc, char **argv)
{
  hsa::init hsa_state;  // Need to destroy this last

  int FD0, FD1, FD2;
  int rc = sscanf(argv[0], "(%d %d %d)", &FD0, &FD1, &FD2);
  if (rc != 3)
    {
      printf("failed to sscanf\n");
      // probably didn't pass file handles in argv[0] then
      return main_with_hsa(argc, argv, nullptr);
    }
  else
    {
      printf("Going to try to run with file handles %d %d %d\n", FD0, FD1, FD2);
      file_handles tmp{FD0, FD1, FD2};

      return main_with_hsa(argc - 1, &argv[1], &tmp);
    }

}
