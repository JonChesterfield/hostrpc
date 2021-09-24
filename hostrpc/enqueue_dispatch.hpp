#ifndef ENQUEUE_DISPATCH_HPP_INCLUDED
#define ENQUEUE_DISPATCH_HPP_INCLUDED

#include "hsa_packet.hpp"

#if defined(__AMDGCN__)

#include <stddef.h>
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

#include "platform.hpp"

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

enum
{
  offset_reserved2 = 384 / 8,
};
static constexpr inline uint32_t getlo(uint64_t x)
{
  return static_cast<uint32_t>(x);
}

static constexpr inline uint32_t gethi(uint64_t x)
{
  return static_cast<uint32_t>(x >> 32u);
}

typedef struct hsa_signal_s
{
  /**
   * Opaque handle. The value 0 is reserved.
   */
  uint64_t handle;
} hsa_signal_t;

typedef enum
{
  /**
   * Queue supports multiple producers.
   */
  HSA_QUEUE_TYPE_MULTI = 0,
  /**
   * Queue only supports a single producer.
   */
  HSA_QUEUE_TYPE_SINGLE = 1
} hsa_queue_type_t;

typedef enum __ockl_memory_order_e
{
  __ockl_memory_order_relaxed = __ATOMIC_RELAXED,
  __ockl_memory_order_acquire = __ATOMIC_ACQUIRE,
  __ockl_memory_order_release = __ATOMIC_RELEASE,
  __ockl_memory_order_acq_rel = __ATOMIC_ACQ_REL,
  __ockl_memory_order_seq_cst = __ATOMIC_SEQ_CST,
} __ockl_memory_order;

typedef struct hsa_queue_s
{
  /**
   * Queue type.
   */
  hsa_queue_type_t type;

  /**
   * Queue features mask. This is a bit-field of ::hsa_queue_feature_t
   * values. Applications should ignore any unknown set bits.
   */
  uint32_t features;

#if 1  // def HSA_LARGE_MODEL
  void *base_address;
#elif defined HSA_LITTLE_ENDIAN
  /**
   * Starting address of the HSA runtime-allocated buffer used to store the AQL
   * packets. Must be aligned to the size of an AQL packet.
   */
  void *base_address;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved0;
#else
  uint32_t reserved0;

  void *base_address;
#endif

  /**
   * Signal object used by the application to indicate the ID of a packet that
   * is ready to be processed. The HSA runtime manages the doorbell signal. If
   * the application tries to replace or destroy this signal, the behavior is
   * undefined.
   *
   * If @a type is ::HSA_QUEUE_TYPE_SINGLE the doorbell signal value must be
   * updated in a monotonically increasing fashion. If @a type is
   * ::HSA_QUEUE_TYPE_MULTI, the doorbell signal value can be updated with any
   * value.
   */
  hsa_signal_t doorbell_signal;

  /**
   * Maximum number of packets the queue can hold. Must be a power of 2.
   */
  uint32_t size;
  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;
  /**
   * Queue identifier, which is unique over the lifetime of the application.
   */
  uint64_t id;

} hsa_queue_t;

// AMD Queue Properties.
typedef uint32_t amd_queue_properties32_t;

#define __ALIGNED__(x) __attribute__((aligned(x)))
// AMD Queue.
#define AMD_QUEUE_ALIGN_BYTES 64
#define AMD_QUEUE_ALIGN __ALIGNED__(AMD_QUEUE_ALIGN_BYTES)
typedef struct AMD_QUEUE_ALIGN amd_queue_s
{
  hsa_queue_t hsa_queue;
  uint32_t reserved1[4];
  volatile uint64_t write_dispatch_id;
  uint32_t group_segment_aperture_base_hi;
  uint32_t private_segment_aperture_base_hi;
  uint32_t max_cu_id;
  uint32_t max_wave_id;
  volatile uint64_t max_legacy_doorbell_dispatch_id_plus_1;
  volatile uint32_t legacy_doorbell_lock;
  uint32_t reserved2[9];
  volatile uint64_t read_dispatch_id;
  uint32_t read_dispatch_id_field_base_byte_offset;
  uint32_t compute_tmpring_size;
  uint32_t scratch_resource_descriptor[4];
  uint64_t scratch_backing_memory_location;
  uint64_t scratch_backing_memory_byte_size;
  uint32_t scratch_workitem_byte_size;
  amd_queue_properties32_t queue_properties;
  uint32_t reserved3[2];
  hsa_signal_t queue_inactive_signal;
  uint32_t reserved4[14];
} amd_queue_t;

#define _MANGLE3x(P, N, S) P##_##N##S
#define MANGLE3x(P, N, S) _MANGLE3x(P, N, S)
#define _MANGLE3(P, N, S) P##_##N##_##S
#define MANGLE3(P, N, S) _MANGLE3(P, N, S)
#define OCKL_MANGLE_T(N, T) MANGLE3(__ockl, N, T)

extern "C" void OCKL_MANGLE_T(hsa_signal, store)(hsa_signal_t sig, long value,
                                                 __ockl_memory_order mem_order);

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
  using global_atomic_uint32 =
      __attribute__((address_space(1))) HOSTRPC_ATOMIC(uint32_t);

  using global_atomic_uint64 =
      __attribute__((address_space(1))) HOSTRPC_ATOMIC(uint64_t);
  auto write_dispatch_id = reinterpret_cast<global_atomic_uint64 *>(
      my_queue + offset::write_dispatch_id());
  auto read_dispatch_id = reinterpret_cast<global_atomic_uint64 *>(
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

  __attribute__((address_space(4))) const void *p =
      __builtin_amdgcn_dispatch_ptr();
  uint64_t user_sig;
  __builtin_memcpy(&user_sig, (const unsigned char *)p + offset_reserved2, 8);
  uint32_t uuid = getlo(user_sig);
  uint32_t state = gethi(user_sig);

  if (platform::is_master_lane(platform::active_threads()))
    {
      uint64_t packet_id =
          platform::atomic_fetch_add<uint64_t, __ATOMIC_RELAXED,
                                     __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
              write_dispatch_id, 1);

      if (0)
        printf(
            "Enqueue dispatch uuid %u w/ state %u enqueue-dispatch slot %lu\n",
            uuid, state, packet_id);

      bool full = true;
      while (full)
        {
          // May want to back off more smoothly on full queue
          uint64_t idx =
              platform::atomic_load<uint64_t, __ATOMIC_ACQUIRE,
                                    __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                  read_dispatch_id);
          full = packet_id >= (size + idx);
          platform::sleep_briefly();
        }

      const uint32_t mask = size - 1;

      unsigned char *base_address;
      __builtin_memcpy(&base_address,
                       (char *)my_queue + offset::queue_base_address(),
                       sizeof(void *));

      unsigned char *packet = (base_address) + packet_size * (packet_id & mask);

      if (0)
        printf("GPU: Using queue size %u at 0x%lx, writing to 0x%lx\n", size,
               (uint64_t)base_address, (uint64_t)packet);

#if (__HAVE_ROCR_HEADERS)
      static_assert(packet_size == sizeof(hsa_kernel_dispatch_packet_t), "");
      assert(packet == (char *)((hsa_kernel_dispatch_packet_t *)
                                    my_amd_queue->hsa_queue.base_address +
                                (packet_id & mask)));
#endif

      platform::fence_acquire();  // new

      {
        const global_atomic_uint32 *s = (const global_atomic_uint32 *)(src);
        global_atomic_uint32 *d = (global_atomic_uint32 *)(packet);

        for (unsigned i = 1; i < 16; i++)
          {
            // storing to first four bytes may be a bad move if CP is watching
            // it if that is the problem, can still copy the rest 8bytes at a
            // time
            platform::atomic_store<uint32_t, __ATOMIC_RELAXED,
                                   __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                &d[i],
                platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                      __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
                    &s[i]));
          }
      }

      // __builtin_memcpy(packet, src, packet_size);

      // printf("Kernel at %u\n",__LINE__);
      // dump_kernel(packet);

      func(packet);

      // printf("Kernel at %u\n",__LINE__);
      // dump_kernel(packet);

#if 0
      printf("enqueue_dispatch written packet[%lu]\n",packet_id);
      hsa_packet::dump_kernel(packet);
      if (0) {
        unsigned char * kernel = (unsigned char*)packet;
        for (unsigned i = 0; i < 64; i++)
          {
            printf(" %u", (unsigned) kernel[i]);
            if (((i+1) % 8) == 0) printf(" -");
          }
        printf(" end\n");
      }
#endif

      uint32_t header =
          platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
              (const global_atomic_uint32 *)(src));
      platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                            __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          (const global_atomic_uint32 *)(src));

      // platform::fence_release(); // new

      if (0)
        printf("Respawn %u w/ state %u writing packet_id %lu\n", uuid, state,
               packet_id);

      platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                             __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
          (global_atomic_uint32 *)packet, header);

      platform::fence_release();  // old

      // storing is excitingly architecture specific. Implementing for gfx >=
      // 900 which can write directly to hardware_doorbell_ptr
      // Non-user signals on < 900 have a mailbox structure to write to, most
      // readily accessed by linking against ockl. May just implement inline.

#if 0
      {
       
        auto my_amd_queue = (__attribute__((address_space(1))) amd_queue_t *)my_queue;
        hsa_signal_t sig{my_amd_queue->hsa_queue.doorbell_signal.handle};
        __ockl_hsa_signal_store(sig, packet_id, __ockl_memory_order_release);
      }
#else
      {
        char *doorbell_handle;
        __builtin_memcpy(&doorbell_handle, my_queue + offset::doorbell_signal(),
                         sizeof(uint64_t));

        HOSTRPC_ATOMIC(uint64_t) * hardware_doorbell_ptr;
        __builtin_memcpy(&hardware_doorbell_ptr,
                         doorbell_handle + offset::hardware_doorbell(),
                         sizeof(HOSTRPC_ATOMIC(uint64_t *)));

        if (0)
          printf(
              "Doorbell: uuid %u, write %lu to address 0x%lx, handle at "
              "0x%lx\n",
              uuid, packet_id, (uint64_t)hardware_doorbell_ptr,
              (uint64_t)doorbell_handle);

        // hsa may be using release fence + relaxed store for this
        platform::atomic_store<uint64_t, __ATOMIC_RELEASE,
                               __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
            hardware_doorbell_ptr, packet_id);
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
