#ifndef POOL_INTERFACE_HPP_INCLUDED
#define POOL_INTERFACE_HPP_INCLUDED

#include "detail/platform_detect.hpp"

// Instantiate one of the boilerplate macros with a symbol and a compile time
// integer for the maximum number of threads that can be managed by the pool.
// This will define a class named SYMBOL. Then define:
// uint32_t SYMBOL::run(uint32_t state);
// on the corresponding architecture
// Compile the code containing the boilerplate macro as each target and
// as opencl (todo: maybe not for nvptx)

#define POOL_INTERFACE_BOILERPLATE_HOST(SYMBOL, MAXIMUM) \
  POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM);

#define POOL_INTERFACE_BOILERPLATE_AMDGPU(SYMBOL, MAXIMUM)             \
  POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM); \
  POOL_INTERFACE_GPU_OPENCL_WRAPPERS(SYMBOL);                          \
  POOL_INTERFACE_GPU_C_WRAPPERS(SYMBOL);

#include "pool_interface_macros.hpp"

// The host thread pool has run and the initialize / finalize etc both
// defined as functions that can be called from the host

#if defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM)
#else
#if HOSTRPC_AMDGCN
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM)
#endif
#if HOSTRPC_HOST
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_HOST(SYMBOL, MAXIMUM) \
  struct SYMBOL : public pool_interface::pthread_pool<SYMBOL, MAXIMUM>    \
  {                                                                       \
   private:                                                               \
    using Base = pool_interface::pthread_pool<SYMBOL, MAXIMUM>;           \
                                                                          \
   public:                                                                \
    uint32_t run(uint32_t);                                               \
    static int initialize() { return 0; }                                 \
    static int finalize() { return 0; }                                   \
    static void bootstrap_entry(uint32_t N) { Base::bootstrap(N, 0); };   \
  };
#endif
#endif

// The amdgpu thread pool currently has run, todo is adding the others
// The symbol defined on the host contains all methods except for run
#if defined(__OPENCL_C_VERSION__)
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM)
#else
#if HOSTRPC_AMDGCN
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM) \
  extern "C" void pool_interface_##SYMBOL##_bootstrap_kernel_target(void);  \
  struct SYMBOL : public pool_interface::hsa_pool<SYMBOL, MAXIMUM>          \
  {                                                                         \
   private:                                                                 \
    using Base = pool_interface::hsa_pool<SYMBOL, MAXIMUM>;                 \
                                                                            \
   public:                                                                  \
    uint32_t run(uint32_t);                                                 \
                                                                            \
   private:                                                                 \
    friend void pool_interface_##SYMBOL##_bootstrap_kernel_target();        \
    static void expose_loop_for_bootstrap_implementation()                  \
    {                                                                       \
      instance()->expose_loop_for_bootstrap_implementation();               \
    }                                                                       \
  };
#endif
#if HOSTRPC_HOST
#define POOL_INTERFACE_THREAD_POOL_TYPE_DECLARATION_AMDGPU(SYMBOL, MAXIMUM) \
  struct SYMBOL                                                             \
  {                                                                         \
    static int initialize(hsa::executable& ex, hsa_queue_t* queue);         \
    static int finalize();                                                  \
    static void bootstrap_entry(uint32_t requested);                        \
    static void set_requested(uint32_t requested);                          \
    static void teardown();                                                 \
                                                                            \
   private:                                                                 \
    static gpu_kernel_info set_requested_;                                  \
    static gpu_kernel_info bootstrap_entry_;                                \
    static gpu_kernel_info teardown_;                                       \
    static hsa_signal_t signal_;                                            \
    static hsa_queue_t* queue_;                                             \
  };                                                                        \
  gpu_kernel_info SYMBOL::set_requested_;                                   \
  gpu_kernel_info SYMBOL::bootstrap_entry_;                                 \
  gpu_kernel_info SYMBOL::teardown_;                                        \
  hsa_signal_t SYMBOL::signal_ = {0};                                       \
  hsa_queue_t* SYMBOL::queue_ = nullptr;                                    \
  int SYMBOL::initialize(hsa::executable& ex, hsa_queue_t* queue)           \
  {                                                                         \
    struct                                                                  \
    {                                                                       \
      const char* name;                                                     \
      gpu_kernel_info* field;                                               \
    } const data[] = {                                                      \
        {                                                                   \
            "__device_pool_interface_" #SYMBOL "_set_requested.kd",         \
            &set_requested_,                                                \
        },                                                                  \
        {                                                                   \
            "__device_pool_interface_" #SYMBOL "_bootstrap_entry.kd",       \
            &bootstrap_entry_,                                              \
        },                                                                  \
        {                                                                   \
            "__device_pool_interface_" #SYMBOL "_teardown.kd",              \
            &teardown_,                                                     \
        },                                                                  \
    };                                                                      \
    int rc = 0;                                                             \
    for (unsigned i = 0; i < sizeof(data) / sizeof(data[0]); i++)           \
      {                                                                     \
        rc += initialize_kernel_info(ex, data[i].name, data[i].field);      \
      }                                                                     \
                                                                            \
    if (rc != 0)                                                            \
      {                                                                     \
        return 1;                                                           \
      }                                                                     \
                                                                            \
    queue_ = queue;                                                         \
    if (hsa_signal_create(1, 0, NULL, &signal_) != HSA_STATUS_SUCCESS)      \
      {                                                                     \
        return 1;                                                           \
      }                                                                     \
    return 0;                                                               \
  }                                                                         \
  int SYMBOL::finalize()                                                    \
  {                                                                         \
    if (signal_.handle)                                                     \
      {                                                                     \
        hsa_signal_destroy(signal_);                                        \
      }                                                                     \
    return 0;                                                               \
  }                                                                         \
  void SYMBOL::bootstrap_entry(uint32_t requested)                          \
  {                                                                         \
    gpu_kernel_info& req = bootstrap_entry_;                                \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,  \
                       req.group_segment_fixed_size, queue_, requested,     \
                       requested, {0});                                     \
  }                                                                         \
  void SYMBOL::set_requested(uint32_t requested)                            \
  {                                                                         \
    gpu_kernel_info& req = set_requested_;                                  \
    hsa::launch_kernel(req.symbol_address, req.private_segment_fixed_size,  \
                       req.group_segment_fixed_size, queue_, requested,     \
                       requested, {0});                                     \
  }                                                                         \
  void SYMBOL::teardown()                                                   \
  {                                                                         \
    invoke_teardown(teardown_, set_requested_, signal_, queue_);            \
  }

#endif
#endif

#if !defined(__OPENCL_C_VERSION__)
// none of this works under opencl at present

#include <stdint.h>

#if !defined(__OPENCL_C_VERSION__)
#if HOSTRPC_AMDGCN
#undef printf
#include "hostrpc_printf.h"
#define printf(...) __hostrpc_printf(__VA_ARGS__)
#endif
#endif

#include "detail/platform.hpp"

#include "hsa_packet.hpp"

#if HOSTRPC_HOST
#include <pthread.h>
#include <stdio.h>
#endif

#include "enqueue_dispatch.hpp"

namespace pool_interface
{
constexpr inline uint64_t pack(uint32_t lo, uint32_t hi)
{
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32u);
}

constexpr inline uint32_t getlo(uint64_t x) { return static_cast<uint32_t>(x); }

constexpr inline uint32_t gethi(uint64_t x)
{
  return static_cast<uint32_t>(x >> 32u);
}

static_assert(0 == getlo(pack(0, 0)), "");
static_assert(0 == gethi(pack(0, 0)), "");
static_assert(1 == getlo(pack(1, 0)), "");
static_assert(0 == gethi(pack(1, 0)), "");
static_assert(0 == getlo(pack(0, 1)), "");
static_assert(1 == gethi(pack(0, 1)), "");
static_assert(1 == getlo(pack(1, 1)), "");
static_assert(1 == gethi(pack(1, 1)), "");

static_assert(0 == getlo(pack(0, 0)), "");
static_assert(0 == gethi(pack(0, 0)), "");
static_assert(UINT32_MAX == getlo(pack(UINT32_MAX, 0)), "");
static_assert(0 == gethi(pack(UINT32_MAX, 0)), "");
static_assert(0 == getlo(pack(0, UINT32_MAX)), "");
static_assert(UINT32_MAX == gethi(pack(0, UINT32_MAX)), "");
static_assert(UINT32_MAX == getlo(pack(UINT32_MAX, UINT32_MAX)), "");
static_assert(UINT32_MAX == gethi(pack(UINT32_MAX, UINT32_MAX)), "");

enum
{
  offset_kernarg = 320 / 8,
  offset_reserved2 = 384 / 8,
  offset_signal = 448 / 8,

  offset_userdata =
      offset_reserved2,  // want an 8 byte field that survives respawn
};

template <typename Derived, uint32_t Max, typename Via>
struct api;

template <typename T>
static inline bool print_enabled(T active_threads)
{
  return true && platform::is_master_lane(active_threads);
}

// Common implementation for thread pools
// pthread, hsa implement
template <uint32_t Max, typename Implementation>
struct threads_base
{
  friend Implementation;
  // Implementation to implement
  // uint32_t get_current_uuid();
  // uint32_t get_stored_state();
  // bool respawn_self(uint32_t /* current state */);
  // int spawn_with_uuid(uint32_t uuid);
  // uint32_t run(uint32_t); // uint32_t threaded through the calls
  // void bootstrap(uint32_t req, const unsigned char* descriptor);
  // void teardown();
  //
  Implementation& implementation()
  {
    return *static_cast<Implementation*>(this);
  }

  constexpr uint32_t maximum() const { return Max; }

  void spawn()
  {
    auto active_threads = platform::active_threads();
    uint32_t uuid = allocate(active_threads);
    if (uuid < /*requested()*/ maximum())  // maximum should be correct too
      {
        if (print_enabled(active_threads))
          {
            printf("uuid %u: spawning %u\n",
                   Implementation().get_current_uuid(), uuid);
          }
        if (Implementation().spawn_with_uuid(uuid) == 0)
          {
            return;
          }
      }
    else
      {
        if (print_enabled(active_threads))
          {
            printf("uuid %u: %u is over maximum %u, dealloc\n",
                   Implementation().get_current_uuid(), uuid, maximum());
          }
      }
    deallocate(active_threads);
  }

 private:
  // This is not safe to run from outside of the pool
  void loop()
  {
    auto active_threads = platform::active_threads();
    uint32_t state = Implementation().get_stored_state();
  start:;
    uint32_t uuid = Implementation().get_current_uuid();
    uint32_t req = requested();

    if (uuid >= req)
      {
        deallocate(active_threads);
        if (print_enabled(active_threads))
          printf("uuid %u >= %u, deallocate (live %u)\n", uuid, req, alive());
        return;
      }

    const uint32_t a = alive();
    if (a < req)
      {
        if (print_enabled(active_threads))
          printf("uuid %u: alive %u < req %u, spawn\n", uuid, a, req);
        // spawn extra. could spawn multiple extra.
        spawn();
      }

    uint32_t nstate = Implementation().run(state);
    if (0)
      if (print_enabled(active_threads))
        {
          printf("Respawn %u w/ state %u->%u\n", uuid, state, nstate);
        }
    state = nstate;

    if (Implementation().respawn_self(state))
      {
        if (0)
          if (print_enabled(active_threads))
            {
              printf("Respawned %u w/ state %u\n", uuid, state);
            }
        return;
      }
    else
      {
        goto start;
      }
  }

  template <typename T>
  bool bootstrap_work_already_done(T active_threads, uint32_t req)
  {
    // This thread was created by a kernel launch, account for it
    uint32_t uuid = allocate(active_threads);

    // bootstrap uses inline argument to set requested as a convenience
    set_requested(req);

    if ((uuid != 0)     // already a thread running
        || (req == 0))  // changing count to zero
      {
        deallocate(active_threads);
        return true;
      }

    return false;
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

  template <typename T>
  uint32_t allocate(T active_threads)
  {
    uint32_t r = {};
    if (platform::is_master_lane(active_threads))
      {
        r = platform::atomic_fetch_add<uint32_t, __ATOMIC_ACQ_REL,
                                       __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
      }
    r = platform::broadcast_master(active_threads, r);
    return r;
  }

  template <typename T>
  void deallocate(T active_threads)
  {
    if (platform::is_master_lane(active_threads))
      {
        platform::atomic_fetch_sub<uint32_t, __ATOMIC_RELAXED,
                                   __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
      }
  }
};

#if HOSTRPC_HOST
template <typename Derived, uint32_t Max>
struct via_pthreads : public threads_base<Max, via_pthreads<Derived, Max>>
{
 public:
  using base = threads_base<Max, via_pthreads<Derived, Max>>;
  friend base;

  uint32_t get_current_uuid() { return current_uuid; }
  uint32_t get_stored_state() { return 0; }

  bool respawn_self(uint32_t) { return false; }

  int spawn_with_uuid(uint32_t uuid)
  {
    uint64_t s = pack(uuid, 0);
    void* arg;
    __builtin_memcpy(&arg, &s, 8);
    return !!pthread_create_detached(pthread_start_routine, arg);
  }

  uint32_t run(uint32_t x) { return static_cast<Derived*>(this)->run(x); }

  void bootstrap(uint32_t req, const unsigned char*)
  {
    auto active_threads = platform::active_threads();
    if (base::bootstrap_work_already_done(active_threads, req))
      {
        return;
      }

    if (spawn_with_uuid(0) == 0)
      {
        // success
      }
    else
      {
        // currently can't report failure to bootstrap
        base::deallocate(active_threads);
      }
  }

  void teardown()
  {
    do
      {
        base::set_requested(0);
        platform::sleep_briefly();
      }
    while (base::alive() != 0);
  }

 private:
  static thread_local uint32_t current_uuid;

  static void* pthread_start_routine(void* p)
  {
    uint64_t u64;
    __builtin_memcpy(&u64, &p, 8);
    current_uuid = getlo(u64);
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
using pthread_pool = api<Derived, Max, via_pthreads<Derived, Max>>;

#endif

#if HOSTRPC_AMDGCN

__attribute__((always_inline)) static char* get_reserved_addr()
{
  __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
  return (char*)p + offset_userdata;
}

static uint64_t load_from_reserved_addr()
{
  uint64_t tmp;
  __builtin_memcpy(&tmp, get_reserved_addr(), 8);
  return tmp;
}

template <typename Derived, uint32_t Max>
struct via_hsa : public threads_base<Max, via_hsa<Derived, Max>>
{
 public:
  using base = threads_base<Max, via_hsa<Derived, Max>>;
  friend base;

  uint32_t get_current_uuid() { return getlo(load_from_reserved_addr()); }
  uint32_t get_stored_state() { return gethi(load_from_reserved_addr()); }

  bool respawn_self(uint32_t state)
  {
    enqueue_helper(get_current_uuid(), state);
    return true;
  }

  int spawn_with_uuid(uint32_t uuid)
  {
    // may be worth adding a spawn-multiple. Would work by fetch-add N to
    // the count, then spawning N waves that get incrementing values as the
    // uuid, derivable from compiler intrinsics from the dispatch
    // those that overshoot can exit early
    enqueue_helper(uuid, 0);
    return 0;
  }

  uint32_t run(uint32_t x) { return static_cast<Derived*>(this)->run(x); }

  void bootstrap(uint32_t req, const unsigned char* descriptor)
  {
    auto active_threads = platform::active_threads();
    if (base::bootstrap_work_already_done(active_threads, req))
      {
        return;
      }

    // This thread now owns uuid=0

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
    auto active_threads = platform::active_threads();
    __attribute__((address_space(4))) const void* p =
        __builtin_amdgcn_dispatch_ptr();

    uint64_t compl_sig;
    __builtin_memcpy(&compl_sig, (const unsigned char*)p + offset_signal, 8);
    uint64_t user_sig;
    __builtin_memcpy(&user_sig, (const unsigned char*)p + offset_reserved2, 8);
    uint64_t kernarg_sig;
    __builtin_memcpy(&kernarg_sig, (const unsigned char*)p + offset_kernarg, 8);

    if (print_enabled(active_threads))
      {
        printf("Teardown %s L%u (a: %u, k: 0x%lx, u: 0x%lx, s: 0x%lx)\n",
               __func__, __LINE__, base::alive(), kernarg_sig, user_sig,
               compl_sig);
      }

    base::set_requested(0);
    const uint32_t a = base::alive();

    if (a != 0)
      {
        if (print_enabled(active_threads))
          printf(
              "Teardown respawned on alive:%u, kill this instance (hope it had "
              "s: 0, was k: 0x%lx, u: 0x%lx, s: 0x%lx)\n",
              a, kernarg_sig, user_sig, compl_sig);

        enqueue_dispatch((const unsigned char*)p);
        return;
      }

    if (print_enabled(active_threads))
      printf(
          "Teardown did not spawn, continuing k: 0x%lx, u: 0x%lx, s: 0x%lx\n",
          kernarg_sig, user_sig, compl_sig);

    if (print_enabled(active_threads))
      printf(
          "Alive should be zero %s L%u (a: %u, k: 0x%lx, u: 0x%lx, s: 0x%lx)\n",
          __func__, __LINE__, a, kernarg_sig, user_sig, compl_sig);

    // Relaunched repeatedly until alive count hit zero. That means there are N
    // threads running, all moving towards exit, none of which will relaunch
    // themselves.
    // This logic copies a signal out of userdata and puts it in the completion
    // slot to signal the host that we're done. It may be simpler to modify
    // the signal directly.

    // All (pool managed) threads have exited. Teardown self.

    // Read completion signal field. If we have one, return to fire it
    uint64_t tmp;
    __builtin_memcpy(&tmp, (const unsigned char*)p + offset_signal, 8);
    if (tmp != 0)
      {
        if (print_enabled(active_threads))
          printf("Signal 0x%lx in completion, returning\n", tmp);
        return;
      }

#if 0
    if (print_enabled(active_threads)) {  printf("Kernel self:\n");
      hsa_packet::dump_kernel((const unsigned char*)p); }
#endif

      // If we don't have a signal, and there isn't one in userdata, we're
      // polling
#if 0
    uint64_t maybe_signal;
    __builtin_memcpy(&maybe_signal, (const unsigned char*)p + offset_userdata,
                     8);
#else
    using global_atomic_uint64 =
        __attribute__((address_space(1))) HOSTRPC_ATOMIC(uint64_t);

    // could atomic load be miscompiling here?
    uint64_t maybe_signal =
        platform::atomic_load<uint64_t, __ATOMIC_RELAXED,
                              __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(
            (const global_atomic_uint64*)((const unsigned char*)p +
                                          offset_userdata));
#endif

    if (maybe_signal == 0)
      {
        if (print_enabled(active_threads))
          printf("No signal, none in userdata, return\n");
        return;
      }
    else
      {
        if (print_enabled(active_threads))
          printf("Retrieved signal 0x%lx from userdata\n", maybe_signal);
      }

    // If there is one, retrieve it from userdata and relaunch
    auto func = [=](unsigned char* packet) {
      // write the non-zero signal to the completion slot
      if (print_enabled(active_threads))
        printf("Write non-zero signal (0x%lx) to completion\n", maybe_signal);

      // platform::atomic_store<uint64_t, __ATOMIC_RELEASE,
      // __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES>(    (HOSTRPC_ATOMIC(uint64_t)
      // *)(packet+offset_signal), maybe_signal);

      __builtin_memcpy(packet + offset_signal, &maybe_signal, 8);
      // uint64_t zero = 0;
      //__builtin_memcpy(packet + offset_userdata, &zero, 8);
    };
    enqueue_dispatch(func, (const unsigned char*)p);
  }

  static void expose_loop_for_bootstrap_implementation()
  {
    Derived::instance()->loop();
  }

 private:
  void enqueue_helper(uint32_t uuid, uint32_t state)
  {
    __attribute__((address_space(4))) void* p = __builtin_amdgcn_dispatch_ptr();
    auto func = [=](unsigned char* packet) {
      uint64_t tmp = pack(uuid, state);
      __builtin_memcpy(packet + offset_userdata, &tmp, 8);
    };
    enqueue_dispatch(func, (const unsigned char*)p);
  }
};

template <typename Derived, uint32_t Max>
using hsa_pool = api<Derived, Max, via_hsa<Derived, Max>>;

#endif

template <typename Derived, uint32_t Max, typename Via>
struct api : private Via
{
 private:
  friend Derived;
  friend Via;
  using Base = Via;
  static Base* instance()
  {
    // will not fare well on gcn if Derived needs a lock around construction
    static Derived e;
    return static_cast<Base*>(&e);
  }

 public:
  static uint32_t get_current_uuid() { return instance()->get_current_uuid(); }

  static void set_requested(uint32_t x) { instance()->set_requested(x); }

  static uint32_t requested() { return instance()->requested(); }

  static uint32_t alive() { return instance()->alive(); }

  static void spawn() { return instance()->spawn(); }

  static uint32_t run(uint32_t x) { instance()->run(x); }

  static void bootstrap(uint32_t req, const unsigned char* d)
  {
    return instance()->bootstrap(req, d);
  }

  static void teardown() { return instance()->teardown(); }
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

inline int initialize_kernel_info(hsa::executable& ex, const char* name,
                                  gpu_kernel_info* info)
{
  uint64_t symbol_address = ex.get_symbol_address_by_name(name);
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

  fprintf(stderr, "kernel_info %s => addr 0x%lx\n", name, symbol_address);

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

inline void invoke_teardown(gpu_kernel_info teardown,
                            gpu_kernel_info set_requested, hsa_signal_t signal,
                            hsa_queue_t* queue)
{
  const hsa_signal_value_t init = 1;
  hsa_signal_store_screlease(signal, init);

  if (0)
    fprintf(stderr, "Host: Call set req 0. Doorbell 0x%lx\n",
            queue->doorbell_signal.handle);

  // teardown is a barrier packet, set to zero asynchronously first
  // for more predictable performance under load

  if (0)
    hsa::launch_kernel(
        set_requested.symbol_address, set_requested.private_segment_fixed_size,
        set_requested.group_segment_fixed_size, queue, 0, 0, {0});

  fprintf(stderr, "Host: Invoke teardown from host set req 0, signal 0x%lx\n",
          signal.handle);

  bool barrier = true;
  uint64_t userdata = signal.handle;
  using namespace pool_interface;
  hsa::launch_kernel(teardown.symbol_address,
                     teardown.private_segment_fixed_size,
                     teardown.group_segment_fixed_size, queue,
                     (offset_userdata == offset_reserved2) ? userdata : 0,
                     (offset_userdata == offset_kernarg) ? userdata : 0,
                     {0} /*signal*/, barrier);

  wait_for_signal_equal_zero(signal, 50000 /*000000*/);

  fprintf(stderr, "Host: Teardown signal reached zero\n");

  hsa_signal_store_screlease(signal, init);
  hsa::launch_kernel(
      set_requested.symbol_address, set_requested.private_segment_fixed_size,
      set_requested.group_segment_fixed_size, queue, 0, 0, signal, barrier);

  fprintf(stderr, "Host: Waiting for set_req to complete, signal 0x%lx\n",
          signal.handle);

  wait_for_signal_equal_zero(signal, 50000 /*000000*/);

  fprintf(stderr, "Host: Teardown returning\n");
}
}  // namespace
#endif

#endif  // !__OPENCL_C_VERSION__
#endif
