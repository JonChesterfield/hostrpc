#ifndef HOSTRPC_CLIENT_HPP_INCLUDED
#define HOSTRPC_CLIENT_HPP_INCLUDED

#include "common.hpp"
#include <functional>
#include <unistd.h>

// Intend to have call and service working across gcn and x86
// The normal terminology is:
// Client makes a call to the server, which does ome work and sends back a reply

namespace hostrpc
{
void fill_nop(page_t*) {}
void use_nop(page_t*) {}

enum class client_state : uint8_t
{
  // inbox outbox active
  idle_client = 0b000,
  active_thread = 0b001,
  work_available = 0b011,
  unknownA = 0b010,  // Invalid? Would mean work posted, nothing returned yet,
                     // nothing waiting
  unknownB = 0b100,  // waiting for server to garbage collect,, no local thread
  garbage_with_thread = 0b101,  // transient state, 0b100 with local thread
  unknownC = 0b110,             // async call, server
  result_available = 0b111,     // thread waiting
};

const char* str(client_state s)
{
  switch (s)
    {
      case client_state::idle_client:
        return "idle_client";
      case client_state::active_thread:
        return "active_thread";
      case client_state::work_available:
        return "work_available";
      case client_state::unknownA:
        return "unknownA";
      case client_state::unknownB:
        return "unknownB";
      case client_state::garbage_with_thread:
        return "garbage_with_thread";
      case client_state::unknownC:
        return "unknownC";
      case client_state::result_available:
        return "result_available";
    }
}

// if inbox is set and outbox not, we are waiting for the server to collect
// garbage that is, can't claim the slot for a new thread is that a sufficient
// criteria for the slot to be awaiting gc?

template <size_t N, typename S>
struct client
{
  client(const mailbox_t<N>* inbox, mailbox_t<N>* outbox, page_t* buffer,
         S step, std::function<void(page_t*)> fill = fill_nop,
         std::function<void(page_t*)> use = use_nop)

      : inbox(inbox),
        outbox(outbox),
        buffer(buffer),
        step(step),
        fill(fill),
        use(use)
  {
  }

  // template <size_t N>
  struct cache
  {
    cache() = default;

    void dump()
    {
      printf("[%lu] %lu/%lu/%lu: %s\n", slot, i, o, a, str(status()));
    }
    client_state status() { return static_cast<client_state>(concat()); }

    bool is(uint8_t s)
    {
      assert(s < 8);
      bool r = s == concat();
      if (!r) dump();
      return r;
    }
    bool is(client_state s) { return s == status(); }

    void init(uint64_t s)
    {
      slot = s;
      word = index_to_element(s);
      subindex = index_to_subindex(s);
    }

    uint64_t i = 0;
    uint64_t o = 0;
    uint64_t a = 0;

    uint64_t slot = UINT64_MAX;
    uint64_t word = UINT64_MAX;
    uint64_t subindex = UINT64_MAX;

   private:
    uint8_t concat()
    {
      unsigned r = detail::nthbitset64(i, subindex) << 2 |
                   detail::nthbitset64(o, subindex) << 1 |
                   detail::nthbitset64(a, subindex) << 0;
      return static_cast<uint8_t>(r);
    }
  };

  client_state status(uint64_t slot)
  {
    size_t w = index_to_element(slot);
    uint64_t subindex = index_to_subindex(slot);

    uint64_t i = inbox->load_word(w);
    uint64_t o = outbox->load_word(w);
    uint64_t a = active.load_word(w);

    unsigned r = detail::nthbitset64(i, subindex) << 2 |
                 detail::nthbitset64(o, subindex) << 1 |
                 detail::nthbitset64(a, subindex) << 0;

    return static_cast<client_state>(r);
  }

  void dump_state(uint64_t slot)
  {
    printf("slot %lu: %s\n", slot, str(status(slot)));
  }

  size_t spin_until_claimed_slot()
  {
    for (;;)
      {
        size_t slot = active.try_claim_any_empty_slot();
        if (slot != SIZE_MAX)
          {
            step(__LINE__);
            return slot;
          }
      }
  }

  size_t words()
  {
    // todo: constexpr, static assert matches outbox and active
    return inbox->words();
  }

  size_t find_candidate_client_slot(uint64_t w, uint64_t* inbox_word,
                                    uint64_t* outbox_word)
  {
    // find a slot which is currently available

    // active must be clear (no other thread using it)
    // outbox must be clear (no data in use)
    // server must also be clear (otherwise waiting on GC)
    // previous sketch featured inbox and outbox clear if active is clear,
    // as a thread waits on the gpu. Going to require all clear here:
    // Checking inbox means we can miss garbage collection at the end of a
    // synchronous task
    // Checking outbox opens the door to async launch

    uint64_t i = inbox->load_word(w);
    uint64_t o = outbox->load_word(w);
    uint64_t a = active.load_word(w);

    uint64_t some_use = i | o | a;

    uint64_t available = ~some_use;
    if (available != 0)
      {
        *inbox_word = i;
        *outbox_word = o;
        return 64 * w + detail::ctz64(available);
      }

    return SIZE_MAX;
  }

  // return true if no garbage (briefly) during call
  bool try_garbage_collect_word_client(uint64_t w)
  {
    auto c = [](uint64_t i, uint64_t) -> uint64_t { return i; };
    return try_garbage_collect_word<N, decltype(c)>(c, inbox, outbox, &active,
                                                    w);
  }

  void dump_word(uint64_t word)
  {
    uint64_t i = inbox->load_word(word);
    uint64_t o = outbox->load_word(word);
    uint64_t a = active.load_word(word);
    printf("%lu %lu %lu\n", i, o, a);
  }

  // Returns true if it successfully launched the task
  bool rpc_invoke(bool have_continuation)
  {
    step(__LINE__);

    // 0b111 is posted request, waited for it, got it
    // 0b110 is posted request, nothing waited, got one
    // 0b101 is got a result, don't need it, only spun up a thread for cleanup
    // 0b100 is got a result, don't need it
    for (uint64_t w = 0; w < inbox->words(); w++)
      {
        try_garbage_collect_word_client(w);
      }

    step(__LINE__);

    // wave_acquire_slot
    // can only acquire a slot which is 000
    size_t slot = SIZE_MAX;

    cache c;
    uint64_t active_word;
    for (uint64_t w = 0; w < words(); w++)
      {
        uint64_t inbox_word, outbox_word;
        // may need to gc for there to be a slot
        try_garbage_collect_word_client(w);
        slot = find_candidate_client_slot(w, &inbox_word, &outbox_word);
        if (slot != SIZE_MAX)
          {
            if (active.try_claim_empty_slot(slot, &active_word))
              {
                printf("try_claim succeeded\n");
                // found a slot and locked it
                c.i = inbox_word;
                c.o = outbox_word;
                c.a = active_word;
                break;
              }
          }
      }

    if (slot == SIZE_MAX)
      {
        // couldn't get a slot, won't launch
        step(__LINE__);

        // currently getting stuck here with outbox full and
        // only some values in the inbox
        return false;
      }

    c.init(slot);

    assert(c.is(0b001));
    step(__LINE__);

    // wave_populate
    fill(&buffer[slot]);
    step(__LINE__);

    // wave_publish work
    {
      uint64_t o = outbox->claim_slot_returning_updated_word(slot);
      c.o = o;
    }

    assert(c.is(0b011));
    // have seen this assert trigger (once)
    assert(status(slot) == client_state::work_available);

    step(__LINE__);

    // current strategy is drop interest in the slot, then wait for the
    // server to confirm, then drop local thread

    // with a continuation, outbox is cleared before this thread returns
    // otherwise, garbage collection eneds to clear that outbox
    if (have_continuation)
      {
        // wait for H1, result available
        uint64_t loaded;
        while ((*inbox)(slot, &loaded) != 1)
          {
            usleep(1000);
          }

        c.i = loaded;
        assert(c.is(0b111));

        step(__LINE__);
        // call the continuation
        use(&buffer[slot]);

        step(__LINE__);

        // mark the work as no longer in use
        // todo: is it better to leave this for the GC?
        {
          uint64_t o = outbox->release_slot_returning_updated_word(slot);
          c.o = o;
        }

        assert(c.is(0b101));
        step(__LINE__);
      }

    // if we don't have a continuation, would return on 0b010
    // this wouldn't be considered garbage by client as inbox is clear
    // the server gets 0b100, does the work, sets the result to 0b110
    // that is then picked up by the client as 0b110

    // wait for H0, result has been garbage collected by the host
    // todo: want to get rid of this busy spin in favour of deferred collection
    // I think that will need an extra client side bitmap

    // We could wait for inbox[slot] != 0 which indicates the result
    // has been garbage collected, but that stalls the wave waiting for the hose
    // Instead, drop the warp and let the allocator skip occupied inbox slots

    // wave release slot
    step(__LINE__);
    {
      uint64_t a = active.release_slot_returning_updated_word(slot);
      c.a = a;
    }

    if (have_continuation)
      {
        assert(c.is(0b100));
      }
    else
      {
        assert(c.is(0b010));
      }
    return true;
  }

  const mailbox_t<N>* inbox;
  mailbox_t<N>* outbox;
  page_t* buffer;
  S step;
  std::function<void(page_t*)> fill;
  std::function<void(page_t*)> use;
  slot_bitmap<N, __OPENCL_MEMORY_SCOPE_DEVICE> active;
};
}  // namespace hostrpc

#endif
