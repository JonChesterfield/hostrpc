#ifndef ENQUEUE_DISPATCH_HPP_INCLUDED
#define ENQUEUE_DISPATCH_HPP_INCLUDED

#if defined(__AMDGCN__)

#include <stdint.h>

inline void enqueue_dispatch(const unsigned char *src);

inline void enqueue_self()
{
  __attribute__((address_space(4))) void *p = __builtin_amdgcn_dispatch_ptr();
  return enqueue_dispatch((const unsigned char *)p);
}

inline void enqueue_dispatch(const unsigned char *src)
{
  // Somewhat hairy implementation.
  const size_t packet_size = 64;

  // Don't think failure can be reported. The hazard is the queue being full,
  // but the spec says one should wait for a space to open up when it is full.
  // fetch_sub is probably not safe with multiple threads accessing the
  // structure.

  auto my_queue =
      (__attribute__((address_space(1))) char *)__builtin_amdgcn_queue_ptr();

  // Acquire an available packet id
  using global_atomic_uint64 =
      __attribute__((address_space(1))) _Atomic(uint64_t) *;
  auto write_dispatch_id =
      reinterpret_cast<global_atomic_uint64>(my_queue + 56);
  auto read_dispatch_id =
      reinterpret_cast<global_atomic_uint64>(my_queue + 128);

  // Need to get queue->size and queue->base_address to use the packet id
  // May want to pass this in as a _constant
  uint32_t size = *(uint32_t *)((char *)my_queue + 24);

  // Inlined platform::is_master_lane
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id =
      __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));

  // TODO: readfirstlane(lane_id) == lowest_active?
  bool is_master_lane = lane_id == lowest_active;

  if (is_master_lane)
    {
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

      const uint32_t mask = size - 1;
      char *packet = ((char *)my_queue + 8) + (packet_id & mask);

      __builtin_memcpy(packet, src, packet_size);
      __c11_atomic_thread_fence(__ATOMIC_RELEASE);

      // need to do a signal store
      uint64_t doorbell_handle = *(uint64_t *)((char *)my_queue + 16);

      // derived from ockl
      char *ptr = (char *)doorbell_handle;
      // kind is first 8 bytes, then a union containing value in next 8 bytes
      _Atomic(uint64_t) *event_mailbox_ptr = (_Atomic(uint64_t) *)(ptr + 16);
      uint32_t *event_id = (uint32_t *)(ptr + 24);

      // can't handle event_mailbox_ptr == null here
      {
        uint32_t id = *event_id;
        __opencl_atomic_store(event_mailbox_ptr, id, __ATOMIC_RELEASE,
                              __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES);

        __builtin_amdgcn_s_sendmsg(1 | (0 << 4),
                                   __builtin_amdgcn_readfirstlane(id) & 0xff);
      }
    }
}

#endif
#endif
