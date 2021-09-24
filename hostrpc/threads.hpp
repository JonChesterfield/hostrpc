#ifndef THREADS_HPP_INCLUDED
#define THREADS_HPP_INCLUDED

#include "platform.hpp"
#include <stdint.h>

namespace hostrpc
{
namespace threads
{
template <uint32_t Max>
struct ty
{
  // Change number of threads
  void set_requested(uint32_t x)
  {
    if (x > maximum())
      {
        x = maximum();
      }
    platform::atomic_store<uint32_t, __ATOMIC_RELEASE,
                           __OPENCL_MEMORY_SCOPE_DEVICE>(&req, x);
  }

  // Query state
  constexpr uint32_t maximum() const { return Max; }
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

  // Convenience functions
  void spawn(int (*spawn_with_uuid)(uint32_t))
  {
    uint32_t UUID = allocate();
    if (UUID < maximum())
      {
        int rc = spawn_with_uuid(UUID);
        if (rc == 0)
          {
            return;
          }
      }

    deallocate();
  }

  bool innerloop(void (*func)(), int (*spawn_with_uuid)(uint32_t),
                 uint32_t UUID)
  {
    // return true to continue executing
    if (UUID >= requested())
      {
        deallocate();
        return false;
      }

    if (alive() < requested())
      {
        // Could call spawn multiple times here
        spawn(spawn_with_uuid);
      }

    func();
    return true;
  }

  // private:
  HOSTRPC_ATOMIC(uint32_t) live = 0;
  HOSTRPC_ATOMIC(uint32_t) req = 0;

  uint32_t allocate()
  {
    auto active_threads = platform::active_threads();
    uint32_t r = {};
    if (platform::is_master_lane(active_threads))
      {
        r = platform::atomic_fetch_add<uint32_t, __ATOMIC_ACQ_REL,
                                       __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
      }
    r = platform::broadcast_master(active_threads, r);
    return r;
  }

  void deallocate()
  {
    auto active_threads = platform::active_threads();
    if (platform::is_master_lane(active_threads))
      {
        platform::atomic_fetch_sub<uint32_t, __ATOMIC_RELAXED,
                                   __OPENCL_MEMORY_SCOPE_DEVICE>(&live, 1);
      }
  }
};

}  // namespace threads
}  // namespace hostrpc

#endif
