#define VISIBLE __attribute__((visibility("default")))

#if !defined __OPENCL__
#if defined(__AMDGCN__)
extern "C" void __device_persistent_call(void *)
{
  // Implemented by caller
}
#endif
#endif

// Kernel entry
#if defined __OPENCL__
void __device_persistent_kernel_cast(__global void *asargs);
kernel void __device_persistent_kernel(__global void *args)
{
  __device_persistent_kernel_cast(args);
}
#endif

#if !defined __OPENCL__

#if 1
#if !defined(__AMDGCN__)
#include "amd_hsa_queue.h"
#include "hsa.h"
#include <stddef.h>
#include <stdint.h>

template <size_t expect, size_t actual>
static void assert_size_t_equal()
{
  static_assert(expect == actual, "");
}

void check_assumptions()
{
  assert_size_t_equal<sizeof(hsa_queue_t), 40>();
  assert_size_t_equal<sizeof(amd_queue_t), 256>();
  assert_size_t_equal<offsetof(hsa_queue_t, size), 24>();
  assert_size_t_equal<offsetof(hsa_queue_t, base_address), 8>();
  assert_size_t_equal<offsetof(hsa_queue_t, doorbell_signal), 16>();
  assert_size_t_equal<offsetof(amd_queue_t, write_dispatch_id), 56>();
  assert_size_t_equal<offsetof(amd_queue_t, read_dispatch_id), 128>();
}

#else
void check_assumptions() {}
#endif
#endif

#include <stddef.h>
#include <stdint.h>

#include "detail/platform.hpp"
#include "interface.hpp"

struct dispatch_packet  // 64 byte aligned, probably
{
  // suspect the private & group segment size need to be set
  dispatch_packet(uint32_t private_segment_size, uint32_t group_segment_size,
                  uint64_t kernel_object, uint64_t kernarg_address)
      : private_segment_size(private_segment_size),
        group_segment_size(group_segment_size),
        kernel_object(kernel_object),
        kernarg_address(kernarg_address)
  {
    check_assumptions();
  }

  // Header also specifies memory scopes. Zero means none.
  // One clear bit indicates no barrier, i.e. don't wait
  uint16_t header = 2;             /*HSA_PACKET_TYPE_KERNEL_DISPATCH*/
  uint16_t setup = 1;              // gridsize
  uint16_t workgroup_size_x = 64;  // todo: wave32
  uint16_t workgroup_size_y = 1;
  uint16_t workgroup_size_z = 1;
  uint16_t reserved0 = 0;
  uint32_t grid_size_x = 64;
  uint32_t grid_size_y = 1;
  uint32_t grid_size_z = 1;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved1 = 0;
  uint64_t completion_signal = 0;  // no signal
};

struct kernel_args
{
  const dispatch_packet *self;
  _Atomic(uint32_t) * control;
  void *application_args;
  void *my_queue;
};

#if defined(__AMDGCN__)

static void kick_signal(uint64_t handle)
{
  // uses a handle to an amd_signal_t
  // that's a complex type, from which we need 'value' to hit the atomic
  // there's a uint32_t event_id and a uint64_t event_mailbox_ptr
  // try to get this working roughly first to avoid working out how to
  // link in the ockl stuff
  // see hsaqs.cl
  char *ptr = (char *)handle;
  // kind is first 8 bytes, then a union containing value in next 8 bytes
  _Atomic(uint64_t) *event_mailbox_ptr = (_Atomic(uint64_t) *)(ptr + 16);
  uint32_t *event_id = (uint32_t *)(ptr + 24);

  assert(event_mailbox_ptr);  // I don't think this should be null

  if (platform::is_master_lane())
    {
      if (event_mailbox_ptr)
        {
          uint32_t id = *event_id;
          __opencl_atomic_store(event_mailbox_ptr, id, __ATOMIC_RELEASE,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

          __builtin_amdgcn_s_sendmsg(1 | (0 << 4),
                                     __builtin_amdgcn_readfirstlane(id) & 0xff);
        }
    }
}

#if 0
static unsigned char *get_dispatch_ptr()
{
  __attribute__((address_space(4))) void *vq = __builtin_amdgcn_dispatch_ptr();
  unsigned char *q = (unsigned char *)vq;
  return q;
}
#endif

extern "C" void __device_persistent_kernel_call(const dispatch_packet *self,
                                                _Atomic(uint32_t) * control,
                                                void *application_args,
                                                void *my_queue)
{
  // The queue intrinsic returns a pointer to __constant memory, which probably
  // can't be mutated by fetch_add. Certainly refuses to compile.

  // dispatch packet available in addr(4), __builtin_amdgcn_dispatch_ptr();
  if (*control == 0)
    {
      __opencl_atomic_store(control, 1, __ATOMIC_RELEASE,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      return;
    }
  else
    {
      __opencl_atomic_store(control, 2, __ATOMIC_RELEASE,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
      return;
    }

  // TODO: Probably call this N times before the tail call
  // Could return based on the return value of this
  __device_persistent_call(application_args);

  // Acquire an available packet id

  _Atomic(uint64_t) *write_dispatch_id =
      reinterpret_cast<_Atomic(uint64_t) *>((char *)my_queue + 56);
  _Atomic(uint64_t) *read_dispatch_id =
      reinterpret_cast<_Atomic(uint64_t) *>((char *)my_queue + 128);

  // Need to get queue->size and queue->base_address to use the packet id
  // May want to pass this in as a _constant
  uint32_t size = *(uint32_t *)((char *)my_queue + 24);

  uint64_t packet_id = platform::critical<uint64_t>([&]() {
    // all devices because other devices can be using the same queue

    uint64_t packet_id =
        __opencl_atomic_fetch_add(write_dispatch_id, 1, __ATOMIC_RELAXED,
                                  __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

    bool full = true;
    while (full)
      {
        // May want to back off more smoothly on full queue
        uint64_t idx =
            __opencl_atomic_load(read_dispatch_id, __ATOMIC_ACQUIRE,
                                 __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);
        full = packet_id >= (size + idx);
      }
    return packet_id;
  });

  const uint32_t mask = size - 1;
  dispatch_packet *packet =
      (dispatch_packet *)((char *)my_queue + 8) + (packet_id & mask);

  *packet = *self;
  __c11_atomic_thread_fence(__ATOMIC_RELEASE);

  // need to do a signal store

  uint64_t doorbell_handle = *(uint64_t *)((char *)my_queue + 16);
  kick_signal(doorbell_handle);
}

extern "C" void __device_persistent_kernel_cast(
    __attribute__((address_space(1))) void *asargs)
{
  kernel_args *args = (kernel_args *)asargs;
  __device_persistent_kernel_call(args->self, args->control,
                                  args->application_args, args->my_queue);
}

#endif

#if defined(__x86_64__)
#include "catch.hpp"
#include "incbin.h"
#include "launch.hpp"

INCBIN(persistent_kernel_so, "persistent_kernel.gcn.so");

TEST_CASE("persistent_kernel")
{
  hsa::init hsa;
  {
    using namespace hostrpc;

    hsa_agent_t kernel_agent = hsa::find_a_gpu_or_exit();
    auto ex = hsa::executable(kernel_agent, persistent_kernel_so_data,
                              persistent_kernel_so_size);
    CHECK(ex.valid());

    const char *kernel_entry = "__device_persistent_kernel.kd";
    uint64_t kernel_address = ex.get_symbol_address_by_name(kernel_entry);

    uint32_t kernel_private_segment_fixed_size = 0;
    uint32_t kernel_group_segment_fixed_size = 0;

    {
      auto m = ex.get_kernel_info();
      auto it = m.find(std::string(kernel_entry));
      if (it != m.end())
        {
          kernel_private_segment_fixed_size =
              it->second.private_segment_fixed_size;
          kernel_group_segment_fixed_size = it->second.group_segment_fixed_size;
        }
      else
        {
          printf("fatal: get_kernel_info failed\n");
          exit(1);
        }
    }

    printf("Kernel address %lx\n", kernel_address);

    hsa_queue_t *queue;
    {
      hsa_status_t rc = hsa_queue_create(
          kernel_agent /* make the queue on this agent */,
          131072 /* todo: size it, this hardcodes max size for vega20 */,
          HSA_QUEUE_TYPE_MULTI /* baseline */,
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

    hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    hsa_region_t coarse_grained_region =
        hsa::region_coarse_grained(kernel_agent);

    {
      uint64_t fail = reinterpret_cast<uint64_t>(nullptr);
      if (kernarg_region.handle == fail || fine_grained_region.handle == fail ||
          coarse_grained_region.handle == fail)
        {
          fprintf(stderr, "Failed to find allocation region on kernel agent\n");
          exit(1);
        }
    }

    size_t N = 1920;
    hostrpc::gcn_x64_t p(N, fine_grained_region.handle,
                         coarse_grained_region.handle);

    // 4 bytes in its own page isn't ideal
    auto control_state =
        hsa::allocate(fine_grained_region, sizeof(_Atomic(uint32_t)));

    kernel_args example = {
        .self = 0,  // initialized during launch_t::launch_t
        .control = new (control_state.get()) _Atomic(uint32_t),
        .application_args = 0,  // unused for now
        .my_queue = (void *)queue,
    };
    __opencl_atomic_store(example.control, 0, __ATOMIC_RELEASE,
                          __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

    auto initializer = [&](hsa_kernel_dispatch_packet_t *packet,
                           with_implicit_args<kernel_args *> *state) -> void {
      fprintf(stderr, "Called initializer\n");
      uint64_t kernarg_address;
      __builtin_memcpy(&kernarg_address, &state, 8);

      // todo: check field offsets match? can the real type be used?
      static_assert(
          sizeof(dispatch_packet) == sizeof(hsa_kernel_dispatch_packet_t), "");

      // set up the state for future packets
      __builtin_memcpy(&(state->state->application_args), &kernarg_address, 8);

      // state.self needs to put the dispatch_packet on the heap as it refers to
      // itself leaks for now (may be able to use intrinsic)
      hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
      auto heap =
          hsa::allocate(fine_grained_region, sizeof(dispatch_packet)).release();
      dispatch_packet *typed_heap =
          new (reinterpret_cast<dispatch_packet *>(heap)) dispatch_packet(
              kernel_private_segment_fixed_size,
              kernel_group_segment_fixed_size, kernel_address, kernarg_address);

      state->state->self = typed_heap;
      state->state->my_queue = queue;

      // set up the first packet
      __builtin_memcpy(packet, typed_heap, sizeof(dispatch_packet));
    };

    auto l = launch_t<kernel_args>(kernel_agent, queue, example, initializer);
    fprintf(stderr, "Launch instance\n");

    while (__opencl_atomic_load(example.control, __ATOMIC_ACQUIRE,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES) == 0)
      {
        fprintf(stderr, "watching control\n");
        platform::sleep_noexcept(10000000);
      }

    fprintf(stderr, "Instance set control to non-zero\n");
  }
}

#endif

#endif
