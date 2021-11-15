#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"
#include "state_machine.hpp"

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does some work and sends back a
// reply

namespace hostrpc
{
struct fill_nop
{
  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, page_t *) {}
  fill_nop() = default;
  fill_nop(const fill_nop &) = delete;
  fill_nop(fill_nop &&) = delete;
};

struct use_nop
{
  HOSTRPC_ANNOTATE void operator()(hostrpc::port_t, page_t *) {}
  use_nop() = default;
  use_nop(const use_nop &) = delete;
  use_nop(use_nop &&) = delete;
};

enum class client_state : uint8_t
{
  // inbox outbox active
  idle_client = 0b000,
  active_thread = 0b001,
  work_available = 0b011,
  async_work_available = 0b010,
  done_pending_server_gc =
      0b100,  // waiting for server to garbage collect, no local thread
  garbage_with_thread = 0b101,  // transient state, 0b100 with local thread
  done_pending_client_gc =
      0b110,                 // created work, result available, no continuation
  result_available = 0b111,  // thread waiting
};

// if inbox is set and outbox not, we are waiting for the server to collect
// garbage that is, can't claim the slot for a new thread is that a sufficient
// criteria for the slot to be awaiting gc?

#if 1
template <typename WordT, typename SZT, typename Counter = counters::client>
struct client_impl : public state_machine_impl<WordT, SZT, Counter,
                                               message_bitmap<WordT, false>,
                                               message_bitmap<WordT, false>>
{
  using base = state_machine_impl<WordT, SZT, Counter,
                                  message_bitmap<WordT, false>,
                                  message_bitmap<WordT, false>>;
  using typename base::state_machine_impl;

  using Word = typename base::Word;
  using SZ = typename base::SZ;
  using lock_t = typename base::lock_t;
  using inbox_t = typename base::inbox_t;
  using outbox_t = typename base::outbox_t;
  using staging_t = typename base::staging_t;

  HOSTRPC_ANNOTATE client_impl()
    : base ()
  {
  }
  HOSTRPC_ANNOTATE ~client_impl() = default;
  HOSTRPC_ANNOTATE client_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, staging_t staging,
                               page_t *shared_buffer)

    : base (sz, active, inbox, outbox, staging, shared_buffer)
  {
    constexpr size_t client_size = 40;

    // SZ is expected to be zero bytes or a uint
    struct SZ_local : public SZ
    {
      float x;
    };
    // Counter is zero bytes for nop or potentially many
    struct Counter_local : public Counter
    {
      float x;
    };
    constexpr bool SZ_empty = sizeof(SZ_local) == sizeof(float);
    constexpr bool Counter_empty = sizeof(Counter_local) == sizeof(float);

    constexpr size_t SZ_size = hostrpc::round8(SZ_empty ? 0 : sizeof(SZ));
    constexpr size_t Counter_size = Counter_empty ? 0 : sizeof(Counter);

    constexpr size_t total_size = client_size + SZ_size + Counter_size;

    static_assert(sizeof(client_impl) == total_size, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE static void *operator new(size_t, client_impl *p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE client_counters get_counters() { return Counter::get(); }

  template <typename T>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port(T active_threads)
  {
    return base::rpc_open_port_lo(active_threads);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port)
  {
    base::rpc_close_port(active_threads, port);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_until_available(T active_threads,
                                                      port_t port)
  {
    typename base::port_state s;
    base::template rpc_port_wait_until_state<T, base::port_state::either_low_or_high>(active_threads, port, &s);

    if (s == base::port_state::high_values)
      {
        rpc_port_discard_result(active_threads, port);
        base::template rpc_port_wait_until_state<T, base::port_state::low_values>(active_threads, port);
      }
  }

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send(T active_threads, port_t port, Op &&op)
  {
    // If the port has just been opened, we know it is available to
    // submit work to. In general, send might be called while the
    // state machine is elsewhere, so conservatively progress it
    // until the slot is empty.
    // There is a potential bug here if 'use' is being used to
    // reset the state, instead of the server clean, as 'use'
    // is not being called, but that might be deemed a API misuse
    // as the callee could have used recv() explicitly instead of
    // dropping the result
    rpc_port_wait_until_available(active_threads, port);  // expensive
    rpc_port_send_given_available<Op>(active_threads, port, cxx::forward<Op>(op));
  }

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send_given_available(T active_threads,
                                                      port_t port, Op &&op)
  {
    base::template rpc_port_apply_lo(active_threads, port, cxx::forward<Op>(op));
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_for_result(T active_threads, port_t port)
  {
    // assumes output live
    assert(bits::nthbitset(
                           base::staging.load_word(this->size(), index_to_element<Word>(port)),
                           index_to_subindex<Word>(port)));
    base::template rpc_port_wait_until_state<T, base::port_state::high_values>(active_threads, port);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_discard_result(T active_threads, port_t port)
  {
    base::template rpc_port_apply_hi(active_threads, port, [](hostrpc::port_t, page_t*) {});
  }

  template <typename Use, typename T>
  HOSTRPC_ANNOTATE void rpc_port_recv(T active_threads, port_t port, Use &&use)
  {
    rpc_port_wait_for_result(active_threads, port);
    base::template rpc_port_apply_hi(active_threads, port, cxx::forward<Use>(use));
  }

    
};

#else
// enabling counters breaks codegen for amdgcn,
template <typename WordT, typename SZT, typename Counter = counters::client>
struct client_impl : public SZT, public Counter
{
  using Word = WordT;
  using SZ = SZT;
  using slot_type = typename SZ::type;
  using lock_t = lock_bitmap<Word>;
  using inbox_t = message_bitmap<Word, false>;
  using outbox_t = message_bitmap<Word, false>;
  using staging_t = slot_bitmap_device_local<Word>;
  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  HOSTRPC_ANNOTATE slot_type size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE slot_type words() const { return size() / wordBits(); }

  page_t *shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  static_assert(cxx::is_trivially_copyable<page_t *>::value, "");
  static_assert(cxx::is_trivially_copyable<lock_t>::value, "");
  static_assert(cxx::is_trivially_copyable<inbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<outbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<staging_t>::value, "");

  HOSTRPC_ANNOTATE client_impl()
      : SZ{},
        Counter{},
        active{},
        inbox{},
        outbox{},
        staging{}
  {
  }
  HOSTRPC_ANNOTATE ~client_impl() = default;
  HOSTRPC_ANNOTATE client_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, staging_t staging,
                               page_t *shared_buffer)

      : SZ{sz},
        Counter{},
        shared_buffer(shared_buffer),
        active(active),
        inbox(inbox),
        outbox(outbox),
        staging(staging)
  {
    constexpr size_t client_size = 40;

    // SZ is expected to be zero bytes or a uint
    struct SZ_local : public SZ
    {
      float x;
    };
    // Counter is zero bytes for nop or potentially many
    struct Counter_local : public Counter
    {
      float x;
    };
    constexpr bool SZ_empty = sizeof(SZ_local) == sizeof(float);
    constexpr bool Counter_empty = sizeof(Counter_local) == sizeof(float);

    constexpr size_t SZ_size = hostrpc::round8(SZ_empty ? 0 : sizeof(SZ));
    constexpr size_t Counter_size = Counter_empty ? 0 : sizeof(Counter);

    constexpr size_t total_size = client_size + SZ_size + Counter_size;

    static_assert(sizeof(client_impl) == total_size, "");
    static_assert(alignof(client_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE void dump()
  {
#if HOSTRPC_HAVE_STDIO
    fprintf(stderr, "shared_buffer %p\n", shared_buffer);
    fprintf(stderr, "inbox         %p\n", inbox.a);
    inbox.dump(size());
    fprintf(stderr, "outbox        %p\n", outbox.a);
    outbox.dump(size());
    fprintf(stderr, "active        %p\n", active.a);
    active.dump(size());
    fprintf(stderr, "outbox stg    %p\n", staging.a);
    staging.dump(size());
#endif
  }

  HOSTRPC_ANNOTATE static void *operator new(size_t, client_impl *p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE client_counters get_counters() { return Counter::get(); }

  template <typename T>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port(T active_threads);  // port_t::unavailable on failure

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port);  // Require != port_t::unavailable, not already closed

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_until_available(T active_threads,
                                                      port_t port);

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send(T active_threads, port_t port, Op &&op);

  template <typename Op, typename T>
  HOSTRPC_ANNOTATE void rpc_port_send_given_available(T active_threads,
                                                      port_t port, Op &&op);

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_wait_for_result(T active_threads, port_t port);

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_discard_result(T active_threads, port_t port)
  {
    release_slot(active_threads, port);
  }

  template <typename Use, typename T>
  HOSTRPC_ANNOTATE void rpc_port_recv(T active_threads, port_t port, Use &&use)
  {
    // wait for H1, result available
    // if outbox is clear, which is detectable, this will not terminate
    rpc_port_wait_for_result(active_threads, port);

    // call the continuation
    use(port, &shared_buffer[static_cast<uint32_t>(port)]);
    platform::fence_release();
    // mark the work as no longer in use
    // todo: is it better to leave this for the GC?
    // can free ports more lazily by updating the staging outbox and
    // leaving the visible one. In that case the update may be transfered
    // for free, or it may never become visible in which case the server
    // won't realise the port is no longer in use
    rpc_port_discard_result(active_threads, port);
  }

 private:
  HOSTRPC_ANNOTATE uint32_t find_candidate_client_slot(uint32_t w)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    Word a = active.load_word(size, w);
    platform::fence_acquire();

    // inbox == outbox == 0 => available for use
    Word available = ~i & ~o & ~a;

    // 1 0 => garbage waiting on server
    // 1 1 => garbage that client can act on
    // Take those that client can act on and are not locked
    Word garbage_todo = i & o & ~a;

    Word candidate = available | garbage_todo;
    if (candidate != 0)
      {
        return wordBits() * w + bits::ctz(candidate);
      }

    return UINT32_MAX;
  }

  template <typename T>
  HOSTRPC_ANNOTATE void release_slot(T active_threads, port_t slot)
  {
    const uint32_t size = this->size();
    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    // opencl has incomplete support for lambdas, can't pass address of
    // captured variable.
    if (platform::is_master_lane(active_threads))
      {
        staged_release_slot(size, slot, &staging, &outbox, &cas_fail_count,
                            &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(active_threads, cas_fail_count);
    cas_help_count = platform::broadcast_master(active_threads, cas_help_count);
    Counter::garbage_cas_fail(active_threads, cas_fail_count);
    Counter::garbage_cas_help(active_threads, cas_help_count);
  }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = staging.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // true if it successfully made a call, false if no work to do or only gc
  // If there's no continuation, shouldn't require a use_application_state

  // Series of functions called with lock[slot] held. Garbage collect if
  // necessary, use slot if possible, wait & call continuationo if necessary

  template <typename T>
  HOSTRPC_ANNOTATE bool rpc_verify_port_available(T active_threads,
                                                  port_t port) noexcept;

  template <typename T>
  HOSTRPC_ANNOTATE bool result_available(T active_threads, port_t port);
};

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE port_t
client_impl<WordT, SZT, Counter>::rpc_open_port(T active_threads)
{
  const slot_type size = this->size();
  const slot_type words = this->words();
  // 0b111 is posted request, waited for it, got it
  // 0b110 is posted request, nothing waited, got one
  // 0b101 is got a result, don't need it, only spun up a thread for cleanup
  // 0b100 is got a result, don't need it

  // tries each word in sequnce. A cas failing suggests contention, in which
  // case try the next word instead of the next slot
  // may be worth supporting non-zero starting word for cache locality effects

  // the array is somewhat contended - attempt to spread out the load by
  // starting clients off at different points in the array. Doesn't make an
  // observable difference in the current benchmark.

  // if the invoke call performed garbage collection, the word is not
  // known to be contended so it may be worth trying a different slot
  // before trying a different word
  for (uint32_t w = 0; w < words; w++)
    {
      uint32_t slot = find_candidate_client_slot(w);
      if (slot == UINT32_MAX)
        {
          // no slot
          Counter::no_candidate_slot(active_threads);
        }
      else
        {
          uint64_t cas_fail_count = 0;
          if (active.try_claim_empty_slot(active_threads, size, slot,
                                          &cas_fail_count))
            {
              // Success, got the lock.
              Counter::cas_lock_fail(active_threads, cas_fail_count);

              // Test if it is available, e.g. isn't garbage
              if (rpc_verify_port_available(active_threads,
                                            static_cast<port_t>(slot)))
                {
                  // Yep, got it and it's good to go
                  return static_cast<port_t>(slot);
                }
            }
          else
            {
              Counter::missed_lock_on_candidate_slot(active_threads);
            }
        }
    }

  // couldn't get a slot in any word
  return port_t::unavailable;
}

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE void client_impl<WordT, SZT, Counter>::rpc_close_port(
    T active_threads, port_t port)
{
  const uint32_t size = this->size();
  // something needs to release() the buffer element before
  // dropping this lock

  assert(port != port_t::unavailable);
  assert(static_cast<uint32_t>(port) < size);

  if (platform::is_master_lane(active_threads))
    {
      active.release_slot(size, port);
    }
}

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE void
client_impl<WordT, SZT, Counter>::rpc_port_wait_until_available(
    T active_threads, port_t port)
{
  const uint32_t size = this->size();
  const uint32_t w = index_to_element<Word>(port);
  const uint32_t subindex = index_to_subindex<Word>(port);

  Word i = inbox.load_word(size, w);
  Word o = staging.load_word(size, w);

  // current thread assumed to hold lock, thus lock is held
  assert(bits::nthbitset(active.load_word(size, w), subindex));

  platform::fence_acquire();

  bool out = bits::nthbitset(o, subindex);
  bool in = bits::nthbitset(i, subindex);

  // io io io io
  // 00 01 10 11

  if (!in & !out)
    {
      // idle client or active thread
      return;  // ready
    }
  // io io io io
  // -- 01 10 11

  if (!in & out)
    {
      // need to wait for result to be available
      while (!in)
        {
          Word i = inbox.load_word(size, w);
          in = bits::nthbitset(i, subindex);
        }
      platform::fence_acquire();
      assert(in);

      // now in & out
    }
  // io io io io
  // -- -- 10 11

  if (in & out)
    {
      // garbage to do
      release_slot(active_threads, port);
      out = false;  // would be false if reloaded
    }
  // io io io io
  // -- -- 10 --

  if (in & !out)  // always true
    {
      // need to to wait for in to clear
      while (in)
        {
          Word i = inbox.load_word(size, w);
          in = bits::nthbitset(i, subindex);
        }
      platform::fence_acquire();
      return;  // ready
    }
  // io io io io
  // -- -- -- --

  __builtin_unreachable();
}

template <typename WordT, typename SZT, typename Counter>
template <typename Op, typename T>
HOSTRPC_ANNOTATE void client_impl<WordT, SZT, Counter>::rpc_port_send(
    T active_threads, port_t port, Op &&op)
{
  // If the port has just been opened, we know it is available to
  // submit work to. In general, send might be called while the
  // state machine is elsewhere, so conservatively progress it
  // until the slot is empty.
  // There is a potential bug here if 'use' is being used to
  // reset the state, instead of the server clean, as 'use'
  // is not being called, but that might be deemed a API misuse
  // as the callee could have used recv() explicitly instead of
  // dropping the result
  rpc_port_wait_until_available(active_threads, port);  // expensive
  rpc_port_send_given_available<Op>(active_threads, port, cxx::forward<Op>(op));
}

template <typename WordT, typename SZT, typename Counter>
template <typename Fill, typename T>
HOSTRPC_ANNOTATE void
client_impl<WordT, SZT, Counter>::rpc_port_send_given_available(
    T active_threads, port_t port, Fill &&fill)
{
  assert(port != port_t::unavailable);
  const uint32_t size = this->size();

  // wave_populate
  // Fill may have no precondition, in which case this doesn't need to run
  fill(port, &shared_buffer[static_cast<uint32_t>(port)]);

  // wave_publish work
  {
    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    if (platform::is_master_lane(active_threads))
      {
        staged_claim_slot(size, port, &staging, &outbox, &cas_fail_count,
                          &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(active_threads, cas_fail_count);
    cas_help_count = platform::broadcast_master(active_threads, cas_help_count);
    Counter::publish_cas_fail(active_threads, cas_fail_count);
    Counter::publish_cas_help(active_threads, cas_help_count);
  }

  // current strategy is drop interest in the port, then wait for the
  // server to confirm, then drop local thread

  // with a continuation, outbox is cleared before this thread returns
  // otherwise, garbage collection needed to clear that outbox

  // if we don't have a continuation, would return on 0b010
  // this wouldn't be considered garbage by client as inbox is clear
  // the server gets 0b100, does the work, sets the result to 0b110
  // that is then picked up by the client as 0b110

  // wait for H0, result has been garbage collected by the host
  // todo: want to get rid of this busy spin in favour of deferred collection
  // I think that will need an extra client side bitmap

  // We could wait for inbox[port] != 0 which indicates the result
  // has been garbage collected, but that stalls the wave waiting for the host
  // Instead, drop the warp and let the allocator skip occupied inbox ports
}

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE void
client_impl<WordT, SZT, Counter>::rpc_port_wait_for_result(T active_threads,
                                                           port_t port)
{
  // assumes output live
  assert(bits::nthbitset(
      staging.load_word(this->size(), index_to_element<Word>(port)),
      index_to_subindex<Word>(port)));

  while (!result_available(active_threads, port))
    {
      // todo: useful work here?
      Counter::waiting_for_result(active_threads);
      platform::sleep();
    }
  platform::fence_acquire();
}

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE bool
client_impl<WordT, SZT, Counter>::rpc_verify_port_available(
    T active_threads, port_t port) noexcept
{
  assert(port != port_t::unavailable);
  const uint32_t element = index_to_element<Word>(port);
  const uint32_t subindex = index_to_subindex<Word>(port);

  const uint32_t size = this->size();
  Word i = inbox.load_word(size, element);
  Word o = staging.load_word(size, element);
  platform::fence_acquire();

  // Called with a lock. The corresponding port can be:
  //  inbox outbox    state  action outbox'
  //      0      0    avail    work       1
  //      0      1     done    none       -
  //      1      0  garbage    none       -
  //      1      1  garbage   clean       0
  // Inbox true means the result has come back
  // That this lock has been taken means no other thread is
  // waiting for that result

  Word this_port = bits::setnthbit((Word)0, subindex);
  Word garbage_todo = i & o & this_port;
  Word available = ~i & ~o & this_port;

  assert((garbage_todo & available) == 0);  // disjoint

  if (garbage_todo)
    {
      release_slot(active_threads, port);
      return false;
    }

  if (!available)
    {
      Counter::got_lock_after_work_done(active_threads);
      return false;
    }

  // Port is available for use.
  return true;
}

template <typename WordT, typename SZT, typename Counter>
template <typename T>
HOSTRPC_ANNOTATE bool client_impl<WordT, SZT, Counter>::result_available(
    T active_threads, port_t slot)
{
  const uint32_t size = this->size();
  uint32_t got = 0;
  if (platform::is_master_lane(
          active_threads))  // TODO: Probably do this on all lanes instead
    {
      got = inbox.read_bit(size, slot);
    }
  got = platform::broadcast_master(active_threads, got);

  return (got == 1);
}

#endif

template <typename WordT, typename SZT, typename Counter = counters::client>
struct client : public client_impl<WordT, SZT, Counter>
{
  using base = client_impl<WordT, SZT, Counter>;
  using base::client_impl;

  static_assert(cxx::is_trivially_copyable<base>::value, "");

  template <typename T, typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke_async(T active_threads, Fill &&fill) noexcept
  {
    auto ApplyFill = hostrpc::make_apply<Fill>(cxx::forward<Fill>(fill));
    // get a port, send it, don't wait
    port_t port = base::rpc_open_port(active_threads);
    if (port == port_t::unavailable)
      {
        return false;
      }
    base::rpc_port_send(active_threads, port, cxx::move(ApplyFill));
    base::rpc_close_port(active_threads, port);
    return true;
  }

  // rpc_invoke returns true if it successfully launched the task
  // returns false if no slot was available

  // Return after calling use(), i.e. waits for server
  template <typename T, typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(T active_threads, Fill &&fill,
                                   Use &&use) noexcept
  {
    auto ApplyFill = hostrpc::make_apply<Fill>(cxx::forward<Fill>(fill));
    auto ApplyUse = hostrpc::make_apply<Use>(cxx::forward<Use>(use));

    port_t port = base::rpc_open_port(active_threads);
    if (port == port_t::unavailable)
      {
        return false;
      }
    base::rpc_port_send(active_threads, port, cxx::move(ApplyFill));
    base::rpc_port_recv(active_threads, port,
                        cxx::move(ApplyUse));  // wait for result
    base::rpc_close_port(active_threads, port);
    return true;
  }

  // TODO: Probably want one of these convenience functions for each rpc_invoke,
  // but perhaps not on volta

  // Return after calling fill(), i.e. does not wait for server
  template <typename Fill>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill &&fill) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_invoke_async(active_threads, cxx::forward<Fill>(fill));
  }

  template <typename Fill, typename Use>
  HOSTRPC_ANNOTATE bool rpc_invoke(Fill &&f, Use &&u) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_invoke(active_threads, cxx::forward<Fill>(f),
                      cxx::forward<Use>(u));
  }
};

}  // namespace hostrpc

#endif
