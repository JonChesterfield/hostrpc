#ifndef HOSTRPC_SERVER_HPP_INCLUDED
#define HOSTRPC_SERVER_HPP_INCLUDED

#include "common.hpp"
#include "counters.hpp"
#include "platform_detect.hpp"

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

template <typename WordT, typename SZT, typename Counter = counters::server>
struct server_impl : public SZT, public Counter
{
  using Word = WordT;
  using SZ = SZT;
  using lock_t = lock_bitmap<Word>;
  using inbox_t = message_bitmap<Word>;
  using outbox_t = message_bitmap<Word>;
  using staging_t = slot_bitmap_device_local<Word>;

  HOSTRPC_ANNOTATE constexpr size_t wordBits() const
  {
    return 8 * sizeof(Word);
  }
  // may want to rename this, number-slots?
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::N(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  HOSTRPC_ANNOTATE server_impl()
      : SZ{0},
        Counter{},
        shared_buffer(nullptr),
        active{},
        inbox{},
        outbox{},
        staging{}
  {
  }
  HOSTRPC_ANNOTATE ~server_impl() {}
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

  // rpc_handle return true if it handled one task, does not attempt multiple.

  // Requires Operate argument.
  // If passed Clear, will call it on garbage collection
  // If passed location, will use it to round robin across slots

  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate &op, Clear& cl,
                                   uint32_t* location) noexcept
  {
    return rpc_handle_impl<Operate, Clear, true>(op, cl, location);
  }

  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate &&op, Clear&& cl,
                                   uint32_t* location) noexcept
  {
    return rpc_handle_impl<Operate, Clear, true>(op, cl, location);
  }

  
  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate& op, uint32_t* location) noexcept
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    Clear cl;
    return rpc_handle_impl<Operate, Clear, false>(op, cl, location);
  }

  // Default location to always start from zero
  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate &op, Clear& cl) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate, Clear>(op, cl, &location);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE bool rpc_handle(Operate& op) noexcept
  {
    uint32_t location = 0;
    return rpc_handle<Operate>(op, &location);
  }

  template <typename Clear>
  HOSTRPC_ANNOTATE uint32_t rpc_open_port(Clear& cl, uint32_t* location_arg)
  {
    return rpc_open_port_impl<Clear, true>(cl, location_arg);
  }

  HOSTRPC_ANNOTATE uint32_t rpc_open_port(uint32_t* location_arg)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    Clear cl;
    return rpc_open_port_impl<Clear, false>(cl, location_arg);
  }

  // default location_arg to zero and discard returned hint
  template <typename Clear>
  HOSTRPC_ANNOTATE uint32_t rpc_open_port(Clear& cl)
  {
    uint32_t location_arg = 0;
    return rpc_open_port<Clear>(cl, &location_arg);
  }

  HOSTRPC_ANNOTATE uint32_t rpc_open_port()
  {
    uint32_t location_arg = 0;
    return rpc_open_port(&location_arg);
  }

  HOSTRPC_ANNOTATE void rpc_close_port(
      uint32_t port)  // Require != UINT32_MAX, not already closed
  {
    const uint32_t size = this->size();
    // something needs to release() the buffer element before
    // dropping this lock

    assert(port != UINT32_MAX);
    assert(port < size);

    if (platform::is_master_lane())
      {
        active.release_slot(size, port);
      }
  }

  template <typename Clear>
  void rpc_port_wait_until_available(uint32_t port, Clear &cl)
  {
    rpc_port_wait_until_available_impl<Clear, true>(port, cl);
  }
  void rpc_port_wait_until_available(uint32_t port)
  {
    struct Clear
    {
      HOSTRPC_ANNOTATE void operator()(uint32_t, hostrpc::page_t*){};
    };
    Clear cl;
    rpc_port_wait_until_available_impl<Clear, false>(port, cl);
  }

  template <typename Operate, typename Clear>
  HOSTRPC_ANNOTATE void rpc_port_operate(Operate &op, Clear &cl, uint32_t slot)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate slot %u, L%u\n", slot,__LINE__);
#endif
    rpc_port_wait_until_available<Clear>(slot, cl);
    rpc_port_operate_given_available(op, slot);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE void rpc_port_operate(Operate &op, uint32_t slot)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate slot %u, L%u\n", slot,__LINE__);
#endif
    rpc_port_wait_until_available(slot);
    rpc_port_operate_given_available(op, slot);
  }

  HOSTRPC_ANNOTATE void rpc_port_operate_publish_operate_done(uint32_t slot)
  {
    const uint32_t size = this->size();
    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    if (platform::is_master_lane())
      {
        staged_claim_slot(size, slot, &staging, &outbox, &cas_fail_count,
                          &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(cas_fail_count);
    cas_help_count = platform::broadcast_master(cas_help_count);
    Counter::publish_cas_fail(cas_fail_count);
    Counter::publish_cas_help(cas_help_count);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available_nopublish(
      Operate &op, uint32_t slot)
  {
    assert(slot != UINT32_MAX);
#if HOSTRPC_HAVE_STDIO
    // printf("rpc port operate_given_available slot %u, L%u\n", slot,__LINE__);
#endif

    // make the calls
    op(slot, &shared_buffer[slot]);
  }

  template <typename Operate>
  HOSTRPC_ANNOTATE void rpc_port_operate_given_available(Operate &op,
                                                         uint32_t slot)
  {
    rpc_port_operate_given_available_nopublish(op, slot);
    // publish result
    rpc_port_operate_publish_operate_done(slot);

    // leaves outbox live
    assert(lock_held(slot));
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

  template <typename Clear, bool have_precondition>
  HOSTRPC_ANNOTATE void garbage_collect(uint32_t slot, Clear &cl)
  {
#if HOSTRPC_HAVE_STDIO
    // printf("gc slot %u\n", slot);
#endif
    const uint32_t size = this->size();
    if (have_precondition)
      {
        cl(slot, &shared_buffer[slot]);
      }

    platform::fence_release();
    uint64_t cas_fail_count = 0;
    uint64_t cas_help_count = 0;
    if (platform::is_master_lane())
      {
        staged_release_slot(size, slot, &staging, &outbox, &cas_fail_count,
                            &cas_help_count);
      }
    cas_fail_count = platform::broadcast_master(cas_fail_count);
    cas_help_count = platform::broadcast_master(cas_help_count);
    Counter::garbage_cas_fail(cas_fail_count);
    Counter::garbage_cas_help(cas_help_count);
  }

  template <typename Clear, bool have_precondition>
  HOSTRPC_ANNOTATE uint32_t rpc_open_port_impl(Clear &cl, uint32_t* location_arg)
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
            Counter::no_candidate_bitmap();
          }
        while (available != 0)
          {
            uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            uint32_t slot = wordBits() * w + idx;
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(size, slot, &cas_fail_count))
              {
                Counter::cas_lock_fail(cas_fail_count);

                // Got the lock. Is there work to do?
                if (rpc_handle_verify_slot_available<Clear, have_precondition>(
                        cl, slot))
                  {
                    // Success. Got a port and work to do. Aim location_arg at
                    // next slot
                    *location_arg = slot + 1;
                    return slot;
                  }
                else
                  {
                    // Failed, drop the lock before continuing to search
                    if (platform::is_master_lane())
                      {
                        active.release_slot(size, slot);
                      }
                  }
              }
            else
              {
                Counter::missed_lock_on_candidate_bitmap();
              }

            // don't try the same slot repeatedly
            available = bits::clearnthbit(available, idx);
          }

        mask = ~((Word)0);
        Counter::missed_lock_on_word();
      }

    // Nothing hit, may as well go from the same location on the next call
    return UINT32_MAX;
  }

  template <typename Clear, bool have_precondition>
  void rpc_port_wait_until_available_impl(uint32_t port, Clear cl)
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
        garbage_collect<Clear, have_precondition>(port, cl);
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

  template <typename Operate, typename Clear, bool have_precondition>
  HOSTRPC_ANNOTATE bool rpc_handle_impl(Operate &op, Clear &cl,
                                        uint32_t* location_arg) noexcept
  {
    const uint32_t port =
        rpc_open_port_impl<Clear, have_precondition>(cl, location_arg);
    if (port == UINT32_MAX)
      {
        return false;
      }

    rpc_port_operate_given_available<Operate>(op, port);

    rpc_close_port(port);
    return true;
  }

  HOSTRPC_ANNOTATE bool lock_held(uint32_t slot)
  {
    const uint32_t element = index_to_element<Word>(slot);
    const uint32_t subindex = index_to_subindex<Word>(slot);
    return bits::nthbitset(active.load_word(size(), element), subindex);
  }

  template <typename Clear, bool have_precondition>
  HOSTRPC_ANNOTATE bool rpc_handle_verify_slot_available(Clear &cl,
                                                         uint32_t slot)
  {
    assert(slot != UINT32_MAX);

    const uint32_t element = index_to_element<Word>(slot);
    const uint32_t subindex = index_to_subindex<Word>(slot);

    const uint32_t size = this->size();

    Word i = inbox.load_word(size, element);
    Word o = staging.load_word(size, element);
    platform::fence_acquire();

    // Called with a lock. The corresponding slot can be:
    //  inbox outbox    state  action outbox'
    //      0      0     idle    none       -
    //      0      1  garbage collect       0
    //      1      0     work    work       1
    //      1      1  waiting    none       -

    Word this_slot = bits::setnthbit((Word)0, subindex);
    Word work_todo = (i & ~o) & this_slot;
    Word garbage_todo = (~i & o) & this_slot;

    assert((work_todo & garbage_todo) == 0);  // disjoint
    assert(lock_held(slot));

    if (garbage_todo)
      {
        assert((o & this_slot) != 0);
        garbage_collect<Clear, have_precondition>(slot, cl);
        assert(lock_held(slot));
        return false;
      }

    if (!work_todo)
      {
        Counter::got_lock_after_work_done();
        assert(lock_held(slot));
        return false;
      }

    return true;
  }
};

}  // namespace hostrpc
#endif
