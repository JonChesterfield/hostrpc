#ifndef HOSTRPC_STATE_MACHINE_HPP_INCLUDED
#define HOSTRPC_STATE_MACHINE_HPP_INCLUDED

#include "../platform/detect.hpp"
#include "common.hpp"
#include "counters.hpp"
#include "cxx.hpp"

namespace hostrpc
{
  // inbox == outbox == 0:
  //   ready to call on_lo
  //
  // inbox == outbox == 1:
  //   ready to call on_hi
  //
  // inbox != outbox:
  //   waiting on other agent
  
template <typename WordT, typename SZT>
struct state_machine_impl : public SZT
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
  HOSTRPC_ANNOTATE uint32_t size() const { return SZ::value(); }
  HOSTRPC_ANNOTATE uint32_t words() const { return size() / wordBits(); }

  page_t* shared_buffer;
  lock_t active;

  inbox_t inbox;
  outbox_t outbox;
  staging_t staging;

  static_assert(cxx::is_trivially_copyable<page_t *>::value, "");
  static_assert(cxx::is_trivially_copyable<lock_t>::value, "");
  static_assert(cxx::is_trivially_copyable<inbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<outbox_t>::value, "");
  static_assert(cxx::is_trivially_copyable<staging_t>::value, "");
  
  HOSTRPC_ANNOTATE state_machine_impl()
      : SZ{},
        Counter{},
        shared_buffer(nullptr),
        active{},
        inbox{},
        outbox{},
        staging{}
  {
  }
  HOSTRPC_ANNOTATE ~state_machine_impl() = default;
  HOSTRPC_ANNOTATE state_machine_impl(SZ sz, lock_t active, inbox_t inbox,
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

  HOSTRPC_ANNOTATE static void* operator new(size_t, state_machine_impl* p)
  {
    return p;
  }

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

  // -> true if successfully opened port, false if failed
  enum class port_state : uint8_t
    {
     unavailable = 0,
     low_values = 1,
     high_values = 2,
     either_low_or_high = 3,
    };


  // port_t::unavailable on failure
  // scan_from can be any uint32_t, it gets % size of the buffer
  // a reasonable guess is a previously used port_t cast to uint32_t and +1

  // open a port with inbox == outbox == 0
  template <typename T>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port_lo(T active_threads, uint32_t scan_from = 0)
  {
    return rpc_open_port<T, port_state::low_values>(active_threads, scan_from, nullptr);
  }

  // open a port with inbox == outbox == 1
  template <typename T>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port_hi(T active_threads, uint32_t scan_from = 0)
  {
    return rpc_open_port<T, port_state::high_values>(active_threads, scan_from, nullptr);
  }


  // open a port with inbox == outbox, writes which type was opened iff passed a port_state
  template <typename T>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port(T active_threads, uint32_t scan_from = 0, port_state * which = nullptr);
  {
    return rpc_open_port<T, port_state::either_low_or_high>(active_threads, scan_from, which);
  }
  
  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port);  // Require != port_t::unavailable, not already closed


  // Dangerous.
  template <typename T, port_state Req>
  void
  rpc_port_wait_until_state(T active_threads, port_t port);

  

private:
  template <port_state Req>
  static HOSTRPC_ANNOTATE Word available_bitmap(Word i, Word o)
  {
    switch(Req)
      {
      case port_state::low_values:
        {
          return ~i & ~o;
        }
      case port_state::high_values:
        {
          return i & o;
        }
      case port_state::either_low_or_high:
        {
          // both 0 or both 1
          return !(i ^ o);
        }
      case port_state::unavailable:
        {
          // different
          return i ^ o;
        }
      }
  }
  
  template <port_state Req>
  HOSTRPC_ANNOTATE bool is_slot_still_available(uint32_t w, uint32_t idx, port_state * which)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    platform::fence_acquire();
    Word r = available_bitmap<Req>(i, o);
    bool available = bits::nthbitset(r, idx);

    if (Req == port_state::either_low_or_high)
      {
        assert(bits::nthbitset(i, idx) ==
               bits::nthbitset(o, idx));
        if (which != nullptr)
          {
            *which = bits::nthbitset(i, idx) ? port_state::high_values : port_state::low_values;
          }
      }
    return available;
  }

  
  template <port_state Req>
  HOSTRPC_ANNOTATE Word find_candidate_available_bitmap(uint32_t w,
                                                        Word mask)
  {
    const uint32_t size = this->size();
    Word i = inbox.load_word(size, w);
    Word o = staging.load_word(size, w);
    Word a = active.load_word(size, w);
    platform::fence_acquire();

    Word r = available_bitmap<Req>(i, o);
    return r & ~a & mask;
  }
  
  template <typename T, port_state Req>
  HOSTRPC_ANNOTATE port_t
  rpc_open_port(T active_threads, uint32_t scan_from, port_state * which)
  {
    static_assert(Req != port_state::unavailable,"");
    const uint32_t size = this->size();
    const uint32_t words = this->words();

    const uint32_t location = scan_from % size;
    const uint32_t element = index_to_element<Word>(location);
    
    // skip bits in the first word <= subindex
    static_assert((sizeof(Word) == 8) || (sizeof(Word) == 4), "");
    Word mask = (sizeof(Word) == 8)
                    ? detail::setbitsrange64(index_to_subindex<Word>(location),
                                             wordBits() - 1)
                    : detail::setbitsrange32(index_to_subindex<Word>(location),
                                             wordBits() - 1);

    // Tries a few bits in element, then all bits in all the other words, then
    // all bits in element. This sometimes overshoots by less than a word.
    // Simpler control flow to check a few values twice when almost none available.
    for (uint32_t wc = 0; wc < words + 1; wc++)
      {
        uint32_t w = (element + wc) % words;
        Word available = find_candidate_available_bitmap<Req>(w, mask);
        if (available == 0)
          {
            // Counter::no_candidate_bitmap(active_threads);
          }
        while (available != 0)
          {
            // tries each bit in incrementing order, clearing them on failure            
            const uint32_t idx = bits::ctz(available);
            assert(bits::nthbitset(available, idx));
            const uint32_t slot = wordBits() * w + idx;
            uint64_t cas_fail_count = 0;
            if (active.try_claim_empty_slot(active_threads, size, slot,
                                            &cas_fail_count))
              {
                // Counter::cas_lock_fail(active_threads, cas_fail_count);

                // Got the lock. Is the slot still available?
                if (is_slot_still_available<Req>(w, idx, which))
                  {
                    // Success. Got a port with the right mailbox settings.
                    assert(slot < size);
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
                // Counter::missed_lock_on_candidate_bitmap(active_threads);
              }

            // don't try the same slot repeatedly
            available = bits::clearnthbit(available, idx);

            
          }
        
        mask = ~((Word)0);
        //Counter::missed_lock_on_word(active_threads);
      }

      if ((Req == port_state::either_low_or_high) && (which != nullptr)) { *which = port_state::unavailable; }
    return port_t::unavailable;
  }
  

  template <typename T>
  HOSTRPC_ANNOTATE void rpc_close_port(
      T active_threads,
      port_t port)  // Require != port_t::unavailable
  {
    const uint32_t size = this->size();
    // warning: something needs to release() the buffer element before
    // dropping this lock

    assert(port != port_t::unavailable);
    if (platform::is_master_lane(active_threads))
      {
        active.release_slot(size, port);
      }
  }

  
};

}

#endif
