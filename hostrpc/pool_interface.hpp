#ifndef POOL_INTERFACE_HPP_INCLUDED
#define POOL_INTERFACE_HPP_INCLUDED

#include "detail/platform_detect.hpp"

#include "pool_interface_macros.hpp"

#if HOSTRPC_AMDGCN && !defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM) \
  struct SYMBOL : public pool_interface::hsa_pool<SYMBOL, MAXIMUM>          \
  {                                                                         \
    void run();                                                             \
  }
#else
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM)
#endif

#if HOSTRPC_HOST && !defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM) \
  struct SYMBOL : public pool_interface::pthread_pool<SYMBOL, MAXIMUM>    \
  {                                                                       \
    void run();                                                           \
  }
#else
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM)
#endif

#define POOL_INTERFACE_BOILERPLATE_AMDGPU(SYMBOL, MAXIMUM)             \
  POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM); \
  POOL_INTERFACE_GPU_OPENCL_WRAPPERS(SYMBOL);                          \
  POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL);                               \
  POOL_INTERFACE_HSA_KERNEL_WRAPPERS(SYMBOL)

#define POOL_INTERFACE_BOILERPLATE_HOST(SYMBOL, MAXIMUM) \
  POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM);

#if !defined(__OPENCL_C_VERSION__)
// none of this works under opencl at present

#include <stdint.h>

#include "detail/platform.hpp"

#include "hsa_packet.hpp"

#if HOSTRPC_HOST
#include <pthread.h>
#include <stdio.h>
#endif

#include "enqueue_dispatch.hpp"

namespace pool_interface
{
#if HOSTRPC_AMDGCN

enum
{
  offset_kernarg = 320 / 8,
  offset_reserved2 = 384 / 8,
  offset_signal = 448 / 8,
};

#endif

template <typename Derived, template <typename, uint32_t> class Via,
          uint32_t Max>
struct api;

template <uint32_t Max, typename Implementation>
struct threads_base
{
  friend Implementation;
  // Implementation to implement
  // uint32_t get_current_uuid();
  // bool respawn_self();
  // int spawn_with_uuid(uint32_t uuid);
  // void run();

  Implementation& implementation()
  {
    return *static_cast<Implementation*>(this);
  }

  constexpr uint32_t maximum() const { return Max; }

  void spawn()
  {
    uint32_t uuid = allocate();
    if (uuid < /*requested()*/ maximum())  // maximum should be correct too
      {
        if (platform::is_master_lane())
          printf("uuid %u: spawning %u\n", Implementation().get_current_uuid(),
                 uuid);
        if (Implementation().spawn_with_uuid(uuid) == 0)
          {
            return;
          }
      }
    deallocate();
  }

 private:
  // This is not safe to run from outside of the pool

  void loop()
  {
  start:;
    uint32_t uuid = Implementation().get_current_uuid();
    uint32_t req = requested();

    if (uuid >= req)
      {
        deallocate();
        if (platform::is_master_lane())
          printf("uuid %u >= %u, deallocate (live %u)\n", uuid, req, alive());
        return;
      }

    uint32_t a = alive();
    if (a < req)
      {
        if (platform::is_master_lane())
          printf("uuid %u: alive %u < req %u, spawn\n", uuid, a, req);
        // spawn extra. could spawn multiple extra.
        spawn();
      }

    Implementation().run();

    if (Implementation().respawn_self())
      {
        return;
      }
    else
      {
        goto start;
      }
  }

  HOSTRPC_ATOMIC(uint32_t) live = 0;
  HOSTRPC_ATOMIC(uint32_t) req = 0;

 public:
  void set_requested(uint32_t x)
  {
    if (x > maximum())
      {
        x = maximum();
      }
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_DEVICE>(&req, x);
  }

  uint32_t alive() const
  {
    return platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                 __OPENCL_MEMORY_SCOPE_DEVICE>(&live);
  }
  uint32_t requested() const
  {
    return platform::atomic_load<uint32_t, __ATOMIC_RELAXED,
                                 __OPENCL_MEMORY_SCOPE_DEVICE>(&req);
  }

  uint32_t allocate()
  {
    uint32_t r = platform::critical<uint32_t>([&]() {
      return platform::atomic_fetch_add<uint32_t, __ATOMIC_ACQ_REL,
                                        __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
    });
    return r;
  }

  void deallocate()
  {
    platform::critical<uint32_t>([&]() {
      return platform::atomic_fetch_sub<uint32_t, __ATOMIC_RELAXED,
                                        __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
    });
  }
};

#if HOSTRPC_HOST
template <typename Derived, uint32_t Max>
struct via_pthreads : public threads_base<Max, via_pthreads<Derived, Max>>
{
 public:
  friend threads_base<Max, via_pthreads<Derived, Max>>;
  uint32_t get_current_uuid() { return current_uuid; }

  bool respawn_self() { return false; }

  int spawn_with_uuid(uint32_t uuid)
  {
    uint64_t s = uuid;
    void* arg;
    __builtin_memcpy(&arg, &s, 8);
    return !!pthread_create_detached(pthread_start_routine, arg);
  }

  void run() { static_cast<Derived*>(this)->run(); }

  void bootstrap_target() { static_cast<Derived*>(this)->loop(); }

  void bootstrap(uint32_t, const unsigned char*)
  {
    // TODO
    __builtin_unreachable();
  }

  void teardown()
  {
    // TODO
    __builtin_unreachable();
  }

 private:
  static thread_local uint32_t current_uuid;

  static void* pthread_start_routine(void* p)
  {
    uint64_t u64;
    __builtin_memcpy(&u64, &p, 8);
    uint32_t u32 = static_cast<uint32_t>(u64);
    current_uuid = u32;

    Derived::instance()->loop();

    return NULL;
  }

  static int pthread_create_detached(void* (*start_routine)(void*), void* arg)
  {
    int rc = 0;
    pthread_attr_t attr;

    rc = pthread_attr_init(&attr);
    if (rc != 0)
      {
        return 1;
      }

    pthread_t handle;
    rc = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (rc != 0)
      {
        (void)pthread_attr_destroy(&attr);
        return 1;
      }

    rc = pthread_create(&handle, NULL, start_routine, arg);
    (void)pthread_attr_destroy(&attr);

    return rc;
  }
};
template <typename Derived, uint32_t Max>
thread_local uint32_t via_pthreads<Derived, Max>::current_uuid;

template <typename Derived, uint32_t Max>
using pthread_pool = api<Derived, via_pthreads, Max>;

#endif

#if HOSTRPC_AMDGCN

__attribute__((always_inline)) static char* get_reserved_addr()
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  return (char*)p + offset_reserved2;
}

static uint32_t load_from_reserved_addr()
{
  uint64_t tmp;
  __builtin_memcpy(&tmp, get_reserved_addr(), 8);
  return (uint32_t)tmp;
}

template <typename Derived, uint32_t Max>
struct via_hsa : public threads_base<Max, via_hsa<Derived, Max>>
{
 public:
  using base = threads_base<Max, via_hsa<Derived, Max>>;
  friend base;

  uint32_t get_current_uuid() { return load_from_reserved_addr(); }

  bool respawn_self()
  {
    enqueue_self();
    return true;
  }

  int spawn_with_uuid(uint32_t uuid)
  {
    // relaunches same kernel that called this
    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
    auto func = [=](unsigned char* packet) {
      uint64_t addr = uuid;
      __builtin_memcpy(packet + offset_reserved2, &addr, 8);
    };

    // copies from p, then calls func
    enqueue_dispatch(func, (const unsigned char*)p);
    return 0;  // succeeds
  }

  void run() { static_cast<Derived*>(this)->run(); }

  void bootstrap_target() { static_cast<Derived*>(this)->loop(); }

  void bootstrap(uint32_t requested, const unsigned char* descriptor)
  {
    // This thread was created by a kernel launch, account for it
    uint32_t uuid = base::allocate();
    if (platform::is_master_lane()) printf("boostrap got uuid %u\n", uuid);

    // bootstrap uses inline argument to set requested as a convenience
    uint32_t req = requested;
    base::set_requested(req);

    if ((uuid != 0)     // already a thread running
        || (req == 0))  // changing count to zero
      {
        base::deallocate();
        return;
      }

    // If the target name is known, e.g. because it is derived from the
    // type name, can avoid passing a kernel in kernarg

    hsa_packet::hsa_kernel_dispatch_packet alternative;
    uint32_t header = hsa_packet::default_header;
    __builtin_memcpy(&alternative, &header, 4);
    hsa_packet::initialize_packet_defaults((unsigned char*)&alternative);
    hsa_packet::write_from_kd_into_hsa(descriptor,
                                       (unsigned char*)&alternative);

    // Read kernel bytes to start new thread (which will be uuid==0)
    enqueue_dispatch((const unsigned char*)&alternative);
  }

  void teardown()
  {
  start:;
    // repeatedly sets zero to win races with threads that spawn more
    // todo: test / prove whether this is aggressive enough, may need to
    // adjust alive or use additional state to ensure termination
    base::set_requested(0);
    uint32_t a = base::alive();

    if (a != 0)
      {
        // wait for the managed threads to exit
        if (respawn_self())
          {
            return;
          }
        else
          {
            goto start;
          }
      }

    // All (pool managed) threads have exited. Teardown self.
    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();

    // Read completion signal field. If we have one, return to fire it
    uint64_t tmp;
    __builtin_memcpy(&tmp, (const unsigned char*)p + offset_signal, 8);
    if (tmp != 0)
      {
        return;
      }

    // If we don't have a signal, and there isn't one in userdata, we're polling
    uint64_t maybe_signal;
    __builtin_memcpy(&maybe_signal, (const unsigned char*)p + offset_reserved2,
                     8);
    if (maybe_signal == 0)
      {
        return;
      }

    // If there is one, retrieve it from userdata and relaunch
    auto func = [=](unsigned char* packet) {
      // write the non-zero signal to the completion slot
      __builtin_memcpy(packet + offset_signal, &maybe_signal, 8);
    };
    enqueue_dispatch(func, (const unsigned char*)p);
  }
};

template <typename Derived, uint32_t Max>
using hsa_pool = api<Derived, via_hsa, Max>;

#endif

template <typename Derived, template <typename, uint32_t> class Via,
          uint32_t Max>
struct api : private Via<Derived, Max>
{
  friend Via<Derived, Max>;
  using Base = Via<Derived, Max>;

  static Derived* instance()
  {
    // will not fare well on gcn if Derived needs a lock around construction
    static Derived e;
    return &e;
  }

  static uint32_t get_current_uuid()
  {
    return static_cast<Base*>(instance())->get_current_uuid();
  }

  static void set_requested(uint32_t x)
  {
    static_cast<Base*>(instance())->set_requested(x);
  }

  static uint32_t requested()
  {
    return static_cast<Base*>(instance())->requested();
  }

  static uint32_t alive() { return static_cast<Base*>(instance())->alive(); }

  static void spawn() { return static_cast<Base*>(instance())->spawn(); }

  static void run() { static_cast<Base*>(instance())->run(); }

  static void bootstrap(uint32_t req, const unsigned char* d)
  {
    return static_cast<Base*>(instance())->bootstrap(req, d);
  }

  static void bootstrap_target()
  {
    return static_cast<Base*>(instance())->bootstrap_target();
  }

  static void teardown() { return static_cast<Base*>(instance())->teardown(); }
};

}  // namespace pool_interface

// In a source file.

#if HOSTRPC_HOST && !defined(__OPENCL_C_VERSION__)

#include "hsa.hpp"

// Kernels launched from the GPU, without reference to any host code,
// presently all use this default header
// TODO: Not ideal to have the static assert on host side for code running
// on gpu side
static_assert(
    hsa_packet::default_header ==
        hsa::packet_header(hsa::header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                           hsa::kernel_dispatch_setup()),
    "");

struct gpu_kernel_info
{
  uint64_t symbol_address = 0;
  uint32_t private_segment_fixed_size = 0;
  uint32_t group_segment_fixed_size = 0;
};

struct hsa_host_pool_state
{
  gpu_kernel_info set_requested;
  gpu_kernel_info bootstrap_entry;
  gpu_kernel_info teardown;
  hsa_signal_t signal = {0};
  hsa_queue_t* queue;
};

namespace
{
inline int initialize_kernel_info(hsa::executable& ex, std::string name,
                                  gpu_kernel_info* info)
{
  uint64_t symbol_address = ex.get_symbol_address_by_name(name.c_str());
  auto m = ex.get_kernel_info();
  auto it = m.find(name);
  if (it == m.end() || symbol_address == 0)
    {
      return 1;
    }
  if ((it->second.private_segment_fixed_size > UINT32_MAX) ||
      (it->second.group_segment_fixed_size > UINT32_MAX))
    {
      return 1;
    }

  info->symbol_address = symbol_address;
  info->private_segment_fixed_size =
      (uint32_t)it->second.private_segment_fixed_size;
  info->group_segment_fixed_size =
      (uint32_t)it->second.group_segment_fixed_size;
  return 0;
}

inline void wait_for_signal_equal_zero(hsa_signal_t signal,
                                       uint64_t timeout_hint = UINT64_MAX)
{
  do
    {
    }
  while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                 timeout_hint, HSA_WAIT_STATE_ACTIVE) != 0);
}

inline void invoke_teardown(gpu_kernel_info teardown, hsa_signal_t signal,
                            hsa_queue_t* queue)
{
  const hsa_signal_value_t init = 1;
  hsa_signal_store_screlease(signal, init);

  hsa::launch_kernel(
      teardown.symbol_address, teardown.private_segment_fixed_size,
      teardown.group_segment_fixed_size, queue, signal.handle, 0, {0});

  wait_for_signal_equal_zero(signal, 50000 /*000000*/);
}
}  // namespace
#endif

#endif  // !__OPENCL_C_VERSION__
#endif
