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
  void loop()
  {
  start:;
    uint32_t uuid = Implementation().get_current_uuid();
    if (uuid >= requested())
      {
        deallocate();
        return;
      }

    if (alive() < requested())
      {
        // spawn extra. could spawn multiple extra.
        spawn();
      }

    Implementation().run();

    bool r = Implementation().respawn_self();
    if (r)
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

#include "enqueue_dispatch.hpp"

template <typename Derived, uint32_t Max>
struct via_hsa : public threads_base<Max, via_hsa<Derived, Max>>
{
 public:
  friend threads_base<Max, via_hsa<Derived, Max>>;

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

 private:
  __attribute__((always_inline)) static char* get_reserved_addr()
  {
    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
    return (char*)p + 48;
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

  static void spawn() { return static_cast<Base*>(instance())->spawn(); }

  static void run() { static_cast<Base*>(instance())->run(); }
};

#if HOSTRPC_HOST
template <typename Derived, uint32_t Max>
using default_pool = pthread_pool<Derived, Max>;
#endif
#if HOSTRPC_AMDGCN
template <typename Derived, uint32_t Max>
using default_pool = hsa_pool<Derived, Max>;
#endif

struct example : public default_pool<example, 16>
{
  void run()
  {
    printf("run from %u\n", get_current_uuid());
    platform::sleep();
  }
};

extern "C"
int main()
{
  example::set_requested(1);

  printf("Hit line %u\n", __LINE__);
  printf("Hit line %u\n", __LINE__);

  example::set_requested(3);
  printf("Hit line %u\n", __LINE__);

  example::spawn();
  printf("Hit line %u\n", __LINE__);

  //  example::loop();
  printf("Hit line %u\n", __LINE__);

  platform::sleep();

  example::set_requested(1);
  platform::sleep();
  example::set_requested(0);
  platform::sleep();

  return 0;
}
