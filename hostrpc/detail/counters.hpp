#ifndef COUNTERS_HPP_INCLUDED
#define COUNTERS_HPP_INCLUDED

#include "platform.hpp"

namespace hostrpc
{
namespace counters
{
// client_nop compiles to no code
// both are default-constructed

struct client
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  client() = default;
  client(const client& o) = default;
  client& operator=(const client& o) = default;

  void no_candidate_slot()
  {
    inc(&state[client_counters::cc_no_candidate_slot]);
  }
  void missed_lock_on_candidate_slot()
  {
    inc(&state[client_counters::cc_missed_lock_on_candidate_slot]);
  }
  void got_lock_after_work_done()
  {
    inc(&state[client_counters::cc_got_lock_after_work_done]);
  }
  void waiting_for_result()
  {
    inc(&state[client_counters::cc_waiting_for_result]);
  }
  void cas_lock_fail(uint64_t c)
  {
    add(&state[client_counters::cc_cas_lock_fail], c);
  }
  void garbage_cas_fail(uint64_t c)
  {
    add(&state[client_counters::cc_garbage_cas_fail], c);
  }
  void publish_cas_fail(uint64_t c)
  {
    add(&state[client_counters::cc_publish_cas_fail], c);
  }
  void finished_cas_fail(uint64_t c)
  {
    // triggers an infinite loop on amdgcn trunk but not amd-stg-open
    add(&state[client_counters::cc_finished_cas_fail], c);
  }

  void garbage_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_garbage_cas_help], c);
  }
  void publish_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_publish_cas_help], c);
  }
  void finished_cas_help(uint64_t c)
  {
    add(&state[client_counters::cc_finished_cas_help], c);
  }

  // client_counters contains non-atomic, const version of this state
  // defined in base_types
  client_counters get()
  {
    __c11_atomic_thread_fence(__ATOMIC_RELEASE);
    client_counters res;
    for (unsigned i = 0; i < client_counters::cc_total_count; i++)
      {
        res.state[i] = state[i];
      }
    return res;
  }

 private:
  _Atomic(uint64_t) state[client_counters::cc_total_count] = {0u};

  static void add(_Atomic(uint64_t) * addr, uint64_t v)
  {
    if (platform::is_master_lane())
      {
        __opencl_atomic_fetch_add(addr, v, __ATOMIC_RELAXED,
                                  __OPENCL_MEMORY_SCOPE_DEVICE);
      }
  }

  static void inc(_Atomic(uint64_t) * addr)
  {
    uint64_t v = 1;
    add(addr, v);
  }
};

struct client_nop
{
  client_nop() {}
  client_counters get() { return {}; }

  void no_candidate_slot() {}
  void missed_lock_on_candidate_slot() {}
  void got_lock_after_work_done() {}
  void waiting_for_result() {}
  void cas_lock_fail(uint64_t) {}

  void garbage_cas_fail(uint64_t) {}
  void publish_cas_fail(uint64_t) {}
  void finished_cas_fail(uint64_t) {}
  void garbage_cas_help(uint64_t) {}
  void publish_cas_help(uint64_t) {}
  void finished_cas_help(uint64_t) {}
};

}  // namespace counters
}  // namespace hostrpc
#endif
