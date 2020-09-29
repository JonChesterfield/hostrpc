#ifndef COUNTERS_HPP_INCLUDED
#define COUNTERS_HPP_INCLUDED

#include "platform.hpp"

namespace hostrpc
{
namespace counters
{
// client_nop compiles to no code
// both are default-constructed
namespace detail
{
template <typename B>
struct client_impl : public B
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  client_impl() = default;
  client_impl(const client_impl& o) = default;
  client_impl& operator=(const client_impl& o) = default;
  void add(unsigned c, uint64_t v) { B::add(c, v); }

  void no_candidate_slot() { inc(client_counters::cc_no_candidate_slot); }
  void missed_lock_on_candidate_slot()
  {
    inc(client_counters::cc_missed_lock_on_candidate_slot);
  }
  void got_lock_after_work_done()
  {
    inc(client_counters::cc_got_lock_after_work_done);
  }
  void waiting_for_result() { inc(client_counters::cc_waiting_for_result); }
  void cas_lock_fail(uint64_t c) { add(client_counters::cc_cas_lock_fail, c); }
  void garbage_cas_fail(uint64_t c)
  {
    add(client_counters::cc_garbage_cas_fail, c);
  }
  void publish_cas_fail(uint64_t c)
  {
    add(client_counters::cc_publish_cas_fail, c);
  }
  void finished_cas_fail(uint64_t c)
  {
    // triggers an infinite loop on amdgcn trunk but not amd-stg-open
    add(client_counters::cc_finished_cas_fail, c);
  }

  void garbage_cas_help(uint64_t c)
  {
    add(client_counters::cc_garbage_cas_help, c);
  }
  void publish_cas_help(uint64_t c)
  {
    add(client_counters::cc_publish_cas_help, c);
  }
  void finished_cas_help(uint64_t c)
  {
    add(client_counters::cc_finished_cas_help, c);
  }

  // client_counters contains non-atomic, const version of this state
  // defined in base_types
  client_counters get()
  {
    __c11_atomic_thread_fence(__ATOMIC_RELEASE);
    client_counters res;
    for (unsigned i = 0; i < client_counters::cc_total_count; i++)
      {
        res.state[i] = B::get(i);
      }
    return res;
  }

 private:
  void inc(unsigned c)
  {
    uint64_t v = 1;
    add(c, v);
  }
};

template <typename B>
struct server_impl : public B
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  server_impl() = default;
  server_impl(const server_impl& o) = default;
  server_impl& operator=(const server_impl& o) = default;
  void add(unsigned c, uint64_t v) { B::add(c, v); }

  void no_candidate_bitmap() { inc(server_counters::sc_no_candidate_bitmap); }

  void cas_lock_fail(uint64_t c) { add(server_counters::sc_cas_lock_fail, c); }

  void missed_lock_on_candidate_bitmap()
  {
    inc(server_counters::sc_missed_lock_on_candidate_bitmap);
  }

  void missed_lock_on_word() { inc(server_counters::sc_missed_lock_on_word); }

  server_counters get()
  {
    __c11_atomic_thread_fence(__ATOMIC_RELEASE);
    server_counters res;
    for (unsigned i = 0; i < server_counters::sc_total_count; i++)
      {
        res.state[i] = B::get(i);
      }
    return res;
  }

 private:
  void inc(unsigned c)
  {
    uint64_t v = 1;
    add(c, v);
  }
};

template <unsigned cap>
struct stateful
{
  uint64_t get(unsigned c) { return state[c]; }
  _Atomic(uint64_t) state[client_counters::cc_total_count] = {0u};

  void add(unsigned c, uint64_t v)
  {
    _Atomic(uint64_t)* addr = &state[c];
    if (platform::is_master_lane())
      {
        __opencl_atomic_fetch_add(addr, v, __ATOMIC_RELAXED,
                                  __OPENCL_MEMORY_SCOPE_DEVICE);
      }
  }
};

struct stateless
{
  uint64_t get(unsigned) { return 0; }
  void add(unsigned, uint64_t) {}
};

}  // namespace detail

using client =
    detail::client_impl<detail::stateful<client_counters::cc_total_count>>;
using client_nop = detail::client_impl<detail::stateless>;

using server =
    detail::server_impl<detail::stateful<server_counters::sc_total_count>>;
using server_nop = detail::server_impl<detail::stateless>;

}  // namespace counters
}  // namespace hostrpc
#endif
