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

#if (__HAVE_ROCR_HEADERS)
#define LITTLEENDIAN_CPU
#include "hsa.h"

#define __GNUC__  // amd_hsa_common uses this to choose attribute((aligned))
#include "amd_hsa_queue.h"
#include "amd_hsa_signal.h"

// if ockl is linked in, can call into the more complicated store machinery
// instead of implementing inline
typedef enum __ockl_memory_order_e
{
  __ockl_memory_order_relaxed = __ATOMIC_RELAXED,
  __ockl_memory_order_acquire = __ATOMIC_ACQUIRE,
  __ockl_memory_order_release = __ATOMIC_RELEASE,
  __ockl_memory_order_acq_rel = __ATOMIC_ACQ_REL,
  __ockl_memory_order_seq_cst = __ATOMIC_SEQ_CST,
} __ockl_memory_order;

extern "C" void __ockl_hsa_signal_store(hsa_signal_t, uint64_t value,
                                        __ockl_memory_order mem_order);

#endif

#include "detail/platform.hpp"

namespace offset
{
inline size_t write_dispatch_id()
{
  constexpr size_t res = 56;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(amd_queue_t, write_dispatch_id), "");
#endif
  return res;
}

inline size_t read_dispatch_id()
{
  constexpr size_t res = 128;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(amd_queue_t, read_dispatch_id), "");
#endif
  return res;
}

inline size_t queue_size()
{
  constexpr size_t res = 24;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(hsa_queue_t, size), "");
#endif
  return res;
}

inline size_t queue_base_address()
{
  constexpr size_t res = 8;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(hsa_queue_t, base_address), "");
#endif
  return res;
}

inline size_t doorbell_signal()
{
  constexpr size_t res = 16;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(hsa_queue_t, doorbell_signal), "");
#endif
  return res;
}

inline size_t hardware_doorbell()
{
  constexpr size_t res = 8;
#if (__HAVE_ROCR_HEADERS)
  static_assert(res == offsetof(amd_signal_t, hardware_doorbell_ptr), "");
#endif
  return res;
}
}  // namespace offset

template <typename F>
inline void enqueue_dispatch(F func, const unsigned char *src)
{
  // Somewhat hairy implementation.
  constexpr size_t packet_size = 64;

  // Don't think failure can be reported. The hazard is the queue being full,
  // but the spec says one should wait for a space to open up when it is full.
  // fetch_sub is probably not safe with multiple threads accessing the
  // structure.

  auto my_queue =
      (__attribute__((address_space(1))) char *)__builtin_amdgcn_queue_ptr();

  // Acquire an available packet id
  using global_atomic_uint64 =
      __attribute__((address_space(1))) HOSTRPC_ATOMIC(uint64_t) *;
  auto write_dispatch_id = reinterpret_cast<global_atomic_uint64>(
      my_queue + offset::write_dispatch_id());
  auto read_dispatch_id = reinterpret_cast<global_atomic_uint64>(
      my_queue + offset::read_dispatch_id());

  // Need to get queue->size and queue->base_address to use the packet id
  // May want to pass this in as a _constant
  uint32_t size;
  {
    auto *s = my_queue + offset::queue_size();
    __builtin_memcpy(&size, s, sizeof(uint32_t));
  }

#if (__HAVE_ROCR_HEADERS)
  auto my_amd_queue = (__attribute__((address_space(1))) amd_queue_t *)my_queue;
  assert((char *)write_dispatch_id == (char *)&my_amd_queue->write_dispatch_id);
  assert((char *)write_dispatch_id == (char *)&my_amd_queue->write_dispatch_id);
  assert(size == my_amd_queue->hsa_queue.size);
#endif

  if (platform::is_master_lane())
    {
      uint64_t packet_id =
          platform::atomic_fetch_add<uint64_t, __ATOMIC_RELAXED,
                                     __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
              write_dispatch_id, 1);

      bool full = true;
      while (full)
        {
          // May want to back off more smoothly on full queue
          uint64_t idx =
              platform::atomic_load<uint64_t, __ATOMIC_ACQUIRE,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                  read_dispatch_id);
          full = packet_id >= (size + idx);
        }

      const uint32_t mask = size - 1;

      unsigned char *base_address;
      __builtin_memcpy(&base_address,
                       (char *)my_queue + offset::queue_base_address(),
                       sizeof(void *));

      unsigned char *packet = (base_address) + packet_size * (packet_id & mask);

#if (__HAVE_ROCR_HEADERS)
      static_assert(packet_size == sizeof(hsa_kernel_dispatch_packet_t), "");
      assert(packet == (char *)((hsa_kernel_dispatch_packet_t *)
                                    my_amd_queue->hsa_queue.base_address +
                                (packet_id & mask)));
#endif

      __builtin_memcpy(packet, src, packet_size);

      func(packet);

#if 0
      printf("enqueue_dispatch written packet:\n");
      dump_kernel(packet);
      {
        unsigned char * kernel = (unsigned char*)packet;
        for (unsigned i = 0; i < 64; i++)
          {
            printf(" %u", (unsigned) kernel[i]);
            if (((i+1) % 8) == 0) printf(" -");
          }
        printf(" end\n");
      }
#endif

      using header_type =
          __attribute__((address_space(1))) HOSTRPC_ATOMIC(uint32_t);

      uint32_t header =
          platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
              (const header_type *)(src));

      platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                             __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          (header_type *)packet, header);

      platform::fence_release();

      // storing is excitingly architecture specific. Implementing for gfx >=
      // 900 which can write directly to hardware_doorbell_ptr
      // Non-user signals on < 900 have a mailbox structure to write to, most
      // readily accessed by linking against ockl. May just implement inline.

#if 1
      {
        char *doorbell_handle;
        __builtin_memcpy(&doorbell_handle, my_queue + offset::doorbell_signal(),
                         sizeof(uint64_t));

        HOSTRPC_ATOMIC(uint64_t) * hardware_doorbell_ptr;
        __builtin_memcpy(&hardware_doorbell_ptr,
                         doorbell_handle + offset::hardware_doorbell(),
                         sizeof(HOSTRPC_ATOMIC(uint64_t *)));

        platform::atomic_store<uint64_t, __ATOMIC_RELEASE,
                               __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
            hardware_doorbell_ptr, packet_id);
      }
#else
      {
        hsa_signal_t sig{my_amd_queue->hsa_queue.doorbell_signal.handle};
        __ockl_hsa_signal_store(sig, packet_id, __ockl_memory_order_release);
      }
#endif
    }
}

inline void enqueue_dispatch(const unsigned char *src)
{
  auto F = [](unsigned char * /*packet*/) {};
  enqueue_dispatch(F, src);
}

#endif
#endif
