#ifndef COUNTERS_HPP_INCLUDED
#define COUNTERS_HPP_INCLUDED

#include "../platform.hpp"

namespace hostrpc
{
namespace counters
{
// client_nop compiles to no code
// both are default-constructed
namespace detail
{
template <typename B>
struct client : public B
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  HOSTRPC_ANNOTATE client() = default;
  HOSTRPC_ANNOTATE client(const client& o) = default;
  HOSTRPC_ANNOTATE client& operator=(const client& o) = default;
  template <typename T>
  HOSTRPC_ANNOTATE void add(T active_threads, unsigned c, uint64_t v)
  {
    B::add(active_threads, c, v);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void no_candidate_slot(T active_threads)
  {
    inc(active_threads, client_counters::cc_no_candidate_slot);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void missed_lock_on_candidate_slot(T active_threads)
  {
    inc(active_threads, client_counters::cc_missed_lock_on_candidate_slot);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void got_lock_after_work_done(T active_threads)
  {
    inc(active_threads, client_counters::cc_got_lock_after_work_done);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void waiting_for_result(T active_threads)
  {
    inc(active_threads, client_counters::cc_waiting_for_result);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void cas_lock_fail(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_cas_lock_fail, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void garbage_cas_fail(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_garbage_cas_fail, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void publish_cas_fail(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_publish_cas_fail, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void finished_cas_fail(T active_threads, uint64_t c)
  {
    // triggers an infinite loop on amdgcn trunk but not amd-stg-open
    add(active_threads, client_counters::cc_finished_cas_fail, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void garbage_cas_help(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_garbage_cas_help, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void publish_cas_help(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_publish_cas_help, c);
  }
  template <typename T>
  HOSTRPC_ANNOTATE void finished_cas_help(T active_threads, uint64_t c)
  {
    add(active_threads, client_counters::cc_finished_cas_help, c);
  }

  // client_counters contains non-atomic, const version of this state
  // defined in base_types
  HOSTRPC_ANNOTATE client_counters get()
  {
    platform::fence_release();
    client_counters res;
    for (unsigned i = 0; i < client_counters::cc_total_count; i++)
      {
        res.state[i] = B::get(i);
      }
    return res;
  }

 private:
  template <typename T>
  HOSTRPC_ANNOTATE void inc(T active_threads, unsigned c)
  {
    uint64_t v = 1;
    add(active_threads, c, v);
  }
};

template <typename B>
struct server : public B
{
  // Probably want this in the interface, partly to keep size
  // lined up (this will be multiple words)
  HOSTRPC_ANNOTATE server() = default;
  HOSTRPC_ANNOTATE server(const server& o) = default;
  HOSTRPC_ANNOTATE server& operator=(const server& o) = default;
  template <typename T>
  HOSTRPC_ANNOTATE void add(T active_threads, unsigned c, uint64_t v)
  {
    B::add(active_threads, c, v);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void no_candidate_bitmap(T active_threads)
  {
    inc(active_threads, server_counters::sc_no_candidate_bitmap);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void cas_lock_fail(T active_threads, uint64_t c)
  {
    add(active_threads, server_counters::sc_cas_lock_fail, c);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void got_lock_after_work_done(T active_threads)
  {
    inc(active_threads, server_counters::sc_got_lock_after_work_done);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void missed_lock_on_candidate_bitmap(T active_threads)
  {
    inc(active_threads, server_counters::sc_missed_lock_on_candidate_bitmap);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void missed_lock_on_word(T active_threads)
  {
    inc(active_threads, server_counters::sc_missed_lock_on_word);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void garbage_cas_fail(T active_threads, uint64_t c)
  {
    add(active_threads, server_counters::sc_garbage_cas_fail, c);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void garbage_cas_help(T active_threads, uint64_t c)
  {
    add(active_threads, server_counters::sc_garbage_cas_help, c);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void publish_cas_fail(T active_threads, uint64_t c)
  {
    add(active_threads, server_counters::sc_publish_cas_fail, c);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void publish_cas_help(T active_threads, uint64_t c)
  {
    add(active_threads, server_counters::sc_publish_cas_help, c);
  }

  HOSTRPC_ANNOTATE server_counters get()
  {
    platform::fence_release();
    server_counters res;
    for (unsigned i = 0; i < server_counters::sc_total_count; i++)
      {
        res.state[i] = B::get(i);
      }
    return res;
  }

 private:
  template <typename T>
  HOSTRPC_ANNOTATE void inc(T active_threads, unsigned c)
  {
    uint64_t v = 1;
    add(active_threads, c, v);
  }
};

template <unsigned cap>
struct stateful
{
  HOSTRPC_ANNOTATE uint64_t get(unsigned c) { return state[c]; }
  HOSTRPC_ATOMIC(uint64_t) state[client_counters::cc_total_count] = {0u};

  template <typename T>
  HOSTRPC_ANNOTATE void add(T active_threads, unsigned c, uint64_t v)
  {
    HOSTRPC_ATOMIC(uint64_t)* addr = &state[c];
    if (platform::is_master_lane(active_threads))
      {
        platform::atomic_fetch_add<uint64_t, __ATOMIC_RELAXED,
                                   __OPENCL_MEMORY_SCOPE_DEVICE>(addr, v);
      }
  }
};

struct stateless
{
  HOSTRPC_ANNOTATE uint64_t get(unsigned) { return 0; }
  template <typename T>
  HOSTRPC_ANNOTATE void add(T, unsigned, uint64_t)
  {
  }
};

}  // namespace detail

using client =
    detail::client<detail::stateful<client_counters::cc_total_count>>;
using client_nop = detail::client<detail::stateless>;

using server =
    detail::server<detail::stateful<server_counters::sc_total_count>>;
using server_nop = detail::server<detail::stateless>;

}  // namespace counters
}  // namespace hostrpc
#endif
