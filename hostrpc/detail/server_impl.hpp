#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"
#include "state_machine.hpp"

namespace hostrpc
{
enum class server_state : uint8_t
{
  // inbox outbox active
  idle_server = 0b000,
  idle_thread = 0b001,
  garbage_available = 0b010,
  garbage_with_thread = 0b011,
  work_available = 0b100,
  work_with_thread = 0b101,
  result_available = 0b110,
  result_with_thread = 0b111,
};

#if 1
template <typename WordT, typename SZT, typename Counter = counters::server>
struct server_impl : public state_machine_impl<WordT, SZT, Counter,
                                               message_bitmap<WordT, true>,
                                               message_bitmap<WordT, false>>
{
  using base =
      state_machine_impl<WordT, SZT, Counter, message_bitmap<WordT, true>,
                         message_bitmap<WordT, false>>;
  using typename base::state_machine_impl;

  using Word = typename base::Word;
  using SZ = typename base::SZ;
  using lock_t = typename base::lock_t;
  using inbox_t = typename base::inbox_t;
  using outbox_t = typename base::outbox_t;
  using staging_t = typename base::staging_t;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  HOSTRPC_ANNOTATE server_impl() : base() {}
  HOSTRPC_ANNOTATE ~server_impl() = default;
  HOSTRPC_ANNOTATE server_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, staging_t staging,
                               page_t* shared_buffer)
      : base(sz, active, inbox, outbox, staging, shared_buffer)
  {
    constexpr size_t server_size = 40;

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

    constexpr size_t total_size = server_size + SZ_size + Counter_size;

    static_assert(sizeof(server_impl) == total_size, "");
    static_assert(alignof(server_impl) == 8, "");
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, server_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE server_counters get_counters() { return Counter::get(); }

  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl,
                                        uint32_t* location_arg)
  {
    return rpc_open_port_impl<Clear, false>(
        active_threads, cxx::forward<Clear>(cl), location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads,
                                        uint32_t* location_arg)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    return rpc_open_port_impl<Clear, false>(active_threads, Clear{},
                                            location_arg);
  }

  // default location_arg to zero and discard returned hint
  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl)
  {
    uint32_t location_arg = 0;
    return rpc_open_port<Clear>(active_threads, cxx::forward<Clear>(cl),
                                &location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads)
  {
    uint32_t location_arg = 0;
    return rpc_open_port(active_threads, &location_arg);
  }

  HOSTRPC_ANNOTATE bool lock_held(port_t port)
  {
    const uint32_t element = index_to_element<Word>(port);
    const uint32_t subindex = index_to_subindex<Word>(port);
    return bits::nthbitset(base::active.load_word(size(), element), subindex);
  }

  template <typename Operate, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available(T active_threads,
                                                         Operate&& op,
                                                         port_t port)
  {
    (void)active_threads;
    (void)op;
    (void)port;

    (void)active_threads;
    assert(port != port_t::unavailable);

    base::rpc_port_apply_lo(active_threads, port, cxx::forward<Operate>(op));

    // claim

    assert(lock_held(port));
  }

 protected:
  template <typename Clear, bool have_precondition, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port_impl(T active_threads, Clear&& cl,
                                             uint32_t* location_arg)
  {
    typename base::port_state ps;
    Clear& clref = cl;

  try_again:;
    port_t p = base::template rpc_open_port(active_threads, *location_arg, &ps);
    if (p == port_t::unavailable)
      {
        return p;
      }

    if (ps == base::port_state::high_values)
      {
        base::rpc_port_apply_hi(active_threads, p, clref);
        base::rpc_close_port(active_threads, p);
        goto try_again;
      }

    assert(ps == base::port_state::low_values);
    *location_arg = static_cast<uint32_t>(p) + 1;
    return p;
  }
};

#else
template <typename WordT, typename SZT, typename Counter = counters::server>
struct server_impl : public SZT, public Counter
{
  using Word = WordT;
  using SZ = SZT;
  using lock_t = lock_bitmap<Word>;
  using inbox_t = message_bitmap<Word, false>;
  using outbox_t = message_bitmap<Word, false>;
  using staging_t = slot_bitmap_device_local<Word>;

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  static_assert(cxx::is_trivially_copyable<page_t*>::value, "");
  static_assert(cxx::is_trivially_copyable<lock_t>::value, "");
  static_assert(cxx::is_trivially_copyable<inbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<outbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<staging_t>::value, "");

  HOSTRPC_ANNOTATE server_impl()
      : SZ{},
        Counter{},
        shared_buffer(nullptr),
        active{},
        inbox{},
        outbox{},
        staging{}
  {
  }
  HOSTRPC_ANNOTATE ~server_impl() = default;
  HOSTRPC_ANNOTATE server_impl(SZ sz, lock_t active, inbox_t inbox,
                               outbox_t outbox, staging_t staging,
                               page_t* shared_buffer)
      : SZ{sz},
        Counter{},
        shared_buffer(shared_buffer),
        active(active),
        inbox(inbox),
        outbox(outbox),
        staging(staging)
  {
  }

  HOSTRPC_ANNOTATE void dump()
  {
#if HOSTRPC_HAVE_STDIO
    fprintf(stderr, "shared_buffer %p\n", shared_buffer);
    fprintf(stderr, "inbox         %p\n", inbox.a);
    fprintf(stderr, "outbox        %p\n", outbox.a);
    fprintf(stderr, "active        %p\n", active.a);
    fprintf(stderr, "outbox stg    %p\n", staging.a);
#endif
  }

  HOSTRPC_ANNOTATE static void* operator new(size_t, server_impl* p)
  {
    return p;
  }

  HOSTRPC_ANNOTATE server_counters get_counters() { return Counter::get(); }

  HOSTRPC_ANNOTATE void dump_word(uint32_t size, Word word)
  {
    Word i = inbox.load_word(size, word);
    Word o = staging.load_word(size, word);
    Word a = active.load_word(size, word);
    (void)(i + o + a);
#if HOSTRPC_HAVE_STDIO
    // printf("%lu %lu %lu\n", i, o, a);
#endif
  }

  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl,
                                        uint32_t* location_arg)
  {
    return rpc_open_port_impl<Clear, true>(
        active_threads, cxx::forward<Clear>(cl), location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads,
                                        uint32_t* location_arg)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    return rpc_open_port_impl<Clear, false>(active_threads, Clear{},
                                            location_arg);
  }

  // default location_arg to zero and discard returned hint
  template <typename Clear, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads, Clear&& cl)
  {
    uint32_t location_arg = 0;
    return rpc_open_port<Clear>(active_threads, cxx::forward<Clear>(cl),
                                &location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port(T active_threads)
  {
    uint32_t location_arg = 0;
    return rpc_open_port(active_threads, &location_arg);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port)  // Require != UINT32_MAX, not already closed
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

  template <typename Clear, typename T>
  void rpc_port_wait_until_available(T active_threads, port_t port, Clear& cl)
  {
    rpc_port_wait_until_available_impl<Clear, true>(active_threads, port, cl);
  }

  template <typename T>
  void rpc_port_wait_until_available(T active_threads, port_t port)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    rpc_port_wait_until_available_impl<Clear, false>(active_threads, port,
                                                     Clear{});
  }

  template <typename Operate, typename Clear, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate(T active_threads, Operate&& op,
                                         Clear&& cl, port_t port)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate slot %u, L%u\n", slot,__LINE__);
#endif
    rpc_port_wait_until_available<Clear>(active_threads, port,
                                         cxx::forward<Clear>(cl));
    rpc_port_operate_given_available(active_threads, cxx::forward<Operate>(op),
                                     port);
  }

  template <typename Operate, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate(T active_threads, Operate&& op,
                                         port_t port)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate slot %u, L%u\n", slot,__LINE__);
#endif
    rpc_port_wait_until_available(active_threads, port);
    rpc_port_operate_given_available(active_threads, cxx::forward<Operate>(op),
                                     port);
  }

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate_publish_operate_done(T active_threads,
                                                              port_t port)
  {
    const uint32_t size = this->size();
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

  template <typename Operate, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available_nopublish(
      T active_threads, Operate&& op, port_t port)
  {
    (void)active_threads;
    assert(port != port_t::unavailable);
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate_given_available slot %u, L%u\n", slot,__LINE__);
#endif

    // make the calls
    op(port, &shared_buffer[static_cast<uint32_t>(port)]);
  }

  template <typename Operate, typename T>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available(T active_threads,
                                                         Operate&& op,
                                                         port_t port)
  {
    rpc_port_operate_given_available_nopublish(active_threads,
                                               cxx::forward<Operate>(op), port);
    // publish result
    rpc_port_operate_publish_operate_done(active_threads, port);

    // leaves outbox live
    assert(lock_held(port));
  }

 private:
  HOSTRPC_ANNOTATE Word find_candidate_server_available_bitmap(uint32_t w,
                                                               Word mask)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    Word a = active.load_word(size, w);
    platform::fence_acquire();

    Word work = i & ~o;
    Word garbage = ~i & o;
    Word todo = work | garbage;
    Word available = todo & ~a & mask;
    return available;
  }

  template <typename Clear, bool have_precondition, typename T>
  HOSTRPC_ANNOTATE void garbage_collect(T active_threads, port_t port, Clear cl)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("gc slot %u\n", slot);
#endif
    const uint32_t size = this->size();
    if (have_precondition)
      {
        cl(port, &shared_buffer[static_cast<uint32_t>(port)]);
      }

    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    if (platform::is_master_lane(active_threads))
      {
        staged_release_slot(size, port, &staging, &outbox, &cas_fail_count,
                            &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(active_threads, cas_fail_count);
    cas_help_count = platform::broadcast_master(active_threads, cas_help_count);
    Counter::garbage_cas_fail(active_threads, cas_fail_count);
    Counter::garbage_cas_help(active_threads, cas_help_count);
  }

 protected:
  template <typename Clear, bool have_precondition, typename T>
  HOSTRPC_ANNOTATE port_t rpc_open_port_impl(T active_threads, Clear&& cl,
                                             uint32_t* location_arg)
  {
    // if an opened port needs garbage collection, does the gc here and
    // continues looking for a port with work to do
    // will return UINT32_MAX if no port with work to do found

    const uint32_t size = this->size();
    const uint32_t words = this->words();

    const uint32_t location = *location_arg % size;
    const uint32_t element = index_to_element<Word>(location);

    // skip bits in the first word <= subindex
    static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");
    Word mask = (sizeof(Word) == 8)
                    ? detail::setbitsrange64(index_to_subindex<Word>(location),
                                             wordBits() - 1)
                    : detail::setbitsrange32(index_to_subindex<Word>(location),
                                             wordBits() - 1);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This overshoots somewhat but ensures that all
    // slots are checked. Could truncate the last word to check each slot
    // exactly once
    for (uint32_t wc = 0; wc < words + 1; wc++)
      {
        uint32_t w = (element + wc) % words;
        Word available = find_candidate_server_available_bitmap(w, mask);
        if (available == 0)
          {
            Counter::no_candidate_bitmap(active_threads);
          }
        while (available != 0)
          {
            uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            uint32_t slot = wordBits() * w + idx;
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(active_threads, size, slot,
                                            &cas_fail_count))
              {
                Counter::cas_lock_fail(active_threads, cas_fail_count);

                // Got the lock. Is there work to do?
                // Can't forward cl here as the call is made repeatedly
                // Will need to refactor verify_slot_available to avoid passing
                // it Until then, force a copy to catch non-copyable clear
                if (rpc_verify_slot_available<Clear, have_precondition>(
                        active_threads, Clear{cl}, static_cast<port_t>(slot)))
                  {
                    // Success. Got a port and work to do. Aim location_arg at
                    // next slot
                    *location_arg = slot + 1;
                    return static_cast<port_t>(slot);
                  }
                else
                  {
                    // Failed, drop the lock before continuing to search
                    if (platform::is_master_lane(active_threads))
                      {
                        active.release_slot(size, static_cast<port_t>(slot));
                      }
                  }
              }
            else
              {
                Counter::missed_lock_on_candidate_bitmap(active_threads);
              }

            // don't try the same slot repeatedly
            available = bits::clearnthbit(available, idx);
          }

        mask = ~((Word)0);
        Counter::missed_lock_on_word(active_threads);
      }

    // Nothing hit, may as well go from the same location on the next call
    return port_t::unavailable;
  }

 private:
  template <typename Clear, bool have_precondition, typename T>
  void rpc_port_wait_until_available_impl(T active_threads, port_t port,
                                          Clear&& cl)
  {
    // port may be:
    // work available
    // garbage available (which will need to call cl)
    // nothing to do
    // work already done

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

#if HOSTRPC_HAVE_STDIO
    // printf("rpc port wait until available slot %u, L%u [%u%u]\n",
    // port,__LINE__,in,out);
#endif

    // io io io io
    // 00 01 10 11

    if (in & out)
      {
        // work already done, inbox will clear before outbox
        while (in)
          {
            Word i = inbox.load_word(size, w);
            in = bits::nthbitset(i, subindex);
          }
        platform::fence_acquire();
      }
    // io io io io
    // 00 01 10 --

    if (!in & out)
      {
        // garbage to do
        garbage_collect<Clear, have_precondition>(active_threads, port, cl);
        out = false;  // would be false if reloaded
      }
    // io io io io
    // 00 -- 10 --

    if (!in & !out)
      {
        // idle, no work to do. Need to wait for work
        while (!in)
          {
            Word i = inbox.load_word(size, w);
            in = bits::nthbitset(i, subindex);
          }
        platform::fence_acquire();
      }
    // io io io io
    // -- -- 10 --

    if (in & !out)
      {
        // work to do
        return;
      }
    // io io io io
    // -- -- -- --

    __builtin_unreachable();
  }

  HOSTRPC_ANNOTATE bool lock_held(port_t port)
  {
    const uint32_t element = index_to_element<Word>(port);
    const uint32_t subindex = index_to_subindex<Word>(port);
    return bits::nthbitset(active.load_word(size(), element), subindex);
  }

  template <typename Clear, bool have_precondition, typename T>
  HOSTRPC_ANNOTATE bool rpc_verify_slot_available(T active_threads, Clear&& cl,
                                                  port_t port)
  {
    const uint32_t size = this->size();
    assert(port != port_t::unavailable);
    const uint32_t element = index_to_element<Word>(port);
    const uint32_t subindex = index_to_subindex<Word>(port);

    Word i = inbox.load_word(size, element);
    Word o = staging.load_word(size, element);
    platform::fence_acquire();

    // Called with a lock. The corresponding port can be:
    //  inbox outbox    state  action outbox'
    //      0      0     idle    none       -
    //      0      1  garbage collect       0
    //      1      0     work    work       1
    //      1      1  waiting    none       -

    Word this_slot = bits::setnthbit((Word)0, subindex);
    Word work_todo = (i & ~o) & this_slot;
    Word garbage_todo = (~i & o) & this_slot;

    assert((work_todo & garbage_todo) == 0);  // disjoint
    assert(lock_held(port));

    if (garbage_todo)
      {
        assert((o & this_slot) != 0);
        garbage_collect<Clear, have_precondition>(active_threads, port, cl);
        assert(lock_held(port));
        return false;
      }

    if (!work_todo)
      {
        Counter::got_lock_after_work_done(active_threads);
        assert(lock_held(port));
        return false;
      }

    return true;
  }
};

#endif

template <typename WordT, typename SZT, typename Counter = counters::server>
struct server : public server_impl<WordT, SZT, Counter>
{
  using base = server_impl<WordT, SZT, Counter>;
  using base::server_impl;
  template <unsigned I, unsigned O>
  using typed_port_t = typename base::template typed_port_t<I, O>;

  static_assert(cxx::is_trivially_copyable<base>::value, "");

  // rpc_handle return true if it handled one task, does not attempt multiple.

  // Requires Operate argument.
  // If passed Clear, will call it on garbage collection
  // If passed location, will use it to round robin across slots

  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, Clear&& cl,
                                   uint32_t* location) noexcept
  {
    auto active_threads = platform::active_threads();
    return rpc_handle_impl<Operate, Clear, true>(
        active_threads, cxx::forward<Operate>(op), cxx::forward<Clear>(cl),
        location);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, uint32_t* location) noexcept
  {
    auto active_threads = platform::active_threads();
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    return rpc_handle_impl<Operate, Clear, false>(
        active_threads, cxx::forward<Operate>(op), Clear{}, location);
  }

  // Default location to always start from zero
  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op, Clear&& cl) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate, Clear>(cxx::forward<Operate>(op),
                                      cxx::forward<Clear>(cl), &location);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate&& op) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate>(cxx::forward<Operate>(op), &location);
  }

 private:
  template <typename T, typename Op, unsigned IandO, bool retValue>
  struct noLambdasInOpenCL
  {
    server* self;
    bool* result;
    Op& op;

    HOSTRPC_ANNOTATE noLambdasInOpenCL(server* self, bool* result, Op& op)
        : self(self), result(result), op(op)
    {
    }

    HOSTRPC_ANNOTATE typed_port_t<IandO, !IandO> operator()(
        T active_threads, typed_port_t<IandO, IandO>&& port)
    {
      typed_port_t<IandO, !IandO> r = self->rpc_port_apply(
          active_threads, cxx::move(port), cxx::forward<Op>(op));
      *result = retValue;
      return r;
    }
  };

  template <typename Operate, typename Clear, bool have_precondition,
            typename T>
  HOSTRPC_ANNOTATE bool rpc_handle_impl(T active_threads, Operate&& op,
                                        Clear&& cl,
                                        uint32_t* location_arg) noexcept
  {
#if 1
    bool result = false;
    // rpc_handle only reports 'true' on operate, garbage collection isn't
    // counted
    auto On00 = noLambdasInOpenCL<T, Operate, 0, true>(this, &result, op);
    auto On11 = noLambdasInOpenCL<T, Clear, 1, false>(this, &result, cl);
    struct None
    {
      /*result initialised to false*/
      HOSTRPC_ANNOTATE void operator()(T) {}
    };

    *location_arg = 1 + base::template rpc_with_opened_port(
                            active_threads, *location_arg, cxx::move(On00),
                            cxx::move(On11), None{});

    return result;

#else

    const port_t port =
        base::template rpc_open_port_impl<Clear, have_precondition>(
            active_threads, cxx::forward<Clear>(cl), location_arg);
    if (port == port_t::unavailable)
      {
        return false;
      }

    base::template rpc_port_operate_given_available<Operate>(
        active_threads, cxx::forward<Operate>(op), port);

    base::rpc_close_port(active_threads, port);
    return true;
#endif
  }
};

}  // namespace hostrpc
#endif
