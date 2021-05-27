#ifndef POOL_INTERFACE_HPP_INCLUDED
#define POOL_INTERFACE_HPP_INCLUDED

#include "detail/platform_detect.hpp"

#include "detail/platform.hpp"

#if HOSTRPC_HOST
#include <pthread.h>
#include <stdio.h>
#endif

#if HOSTRPC_AMDGCN
#undef printf
#include "hostrpc_printf.h"
#endif

#include "dump_kernel.i"

namespace pool_interface
{
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
    if (uuid < maximum())
      {
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
        return;
      }

    if (alive() < req)
      {
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

  void bootstrap(const unsigned char*)
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

static inline uint32_t get_lane_id(void)
{
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
static inline bool is_master_lane(void)
{
  // TODO: 32 wide wavefront, consider not using raw intrinsics here
  uint64_t activemask = __builtin_amdgcn_read_exec();

  // TODO: check codegen for trunc lowest_active vs expanding lane_id
  // TODO: ffs is lifted from openmp runtime, looks like it should be ctz
  uint32_t lowest_active = __builtin_ffsl(activemask) - 1;
  uint32_t lane_id = get_lane_id();

  // TODO: readfirstlane(lane_id) == lowest_active?
  return lane_id == lowest_active;
}

#include "enqueue_dispatch.hpp"

template <typename Derived, uint32_t Max>
struct via_hsa : public threads_base<Max, via_hsa<Derived, Max>>
{
 public:
  using base = threads_base<Max, via_hsa<Derived, Max>>;
  friend base;

  uint32_t get_current_uuid()
  {
    uint64_t res2;
    __builtin_memcpy(&res2, get_reserved_addr(), 8);
    return (uint32_t)res2;
  }

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
      __builtin_memcpy(packet + 40, &addr, 8);
    };

    // copies from p, then calls func
    enqueue_dispatch(func, (const unsigned char*)p);
    return 0;  // succeeds
  }

  void run() { static_cast<Derived*>(this)->run(); }

  void bootstrap_target() { static_cast<Derived*>(this)->loop(); }

  void bootstrap(const unsigned char* kernel)
  {
    uint32_t req = load_from_reserved_addr();
    base::set_requested(req);
    if (req == 0)
      {
        return;
      }
    if (base::alive() != 0)
      {
        return;
      }

    // If none are running, need to start the process

    base::allocate();  // increases live count to 1 for the new thread
    enqueue_dispatch(kernel);
    // return to dispose of this bootstrap kernel
  }

  void teardown()
  {
  start:;
    base::set_requested(0);
    uint32_t a = base::alive();

    if (a != 0)
      {
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

    // Read signal field. If we have one, return to fire it
    uint64_t tmp;
    __builtin_memcpy(&tmp, (const unsigned char*)p + (448 / 8),
                     8);  // read signal slot
    if (tmp != 0)
      {
        return;
      }

    // If we don't have a signal, retrieve it from userdata and relaunch
    auto func = [=](unsigned char* packet) {
      uint64_t tmp;
      // read signal from reserved2 and write it to signal slot
      __builtin_memcpy(&tmp, packet + (384 / 8), 8);
      __builtin_memcpy(packet + (448 / 8), &tmp, 8);
    };
    enqueue_dispatch(func, (const unsigned char*)p);
  }

 private:
  __attribute__((always_inline)) static char* get_reserved_addr()
  {
    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
    return (char*)p + 48;
  }
  static uint32_t load_from_reserved_addr()
  {
    uint64_t tmp;
    __builtin_memcpy(&tmp, get_reserved_addr(), 8);
    return (uint32_t)tmp;
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

  static void bootstrap(const unsigned char* k)
  {
    return static_cast<Base*>(instance())->bootstrap(k);
  }

  static void bootstrap_target()
  {
    return static_cast<Base*>(instance())->bootstrap_target();
  }

  static void teardown() { return static_cast<Base*>(instance())->teardown(); }
};

#if HOSTRPC_HOST
template <typename Derived, uint32_t Max>
using default_pool = pthread_pool<Derived, Max>;
#endif
#if HOSTRPC_AMDGCN
template <typename Derived, uint32_t Max>
using default_pool = hsa_pool<Derived, Max>;
#endif

}  // namespace pool_interface
#endif
