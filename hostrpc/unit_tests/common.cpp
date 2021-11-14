#include "../detail/common.hpp"
#include "EvilUnit.h"

#if defined __AMDGCN__
namespace hostcall_ops
{
// TODO: Clean up this stuff, it's way harder to link in printf than it needs to
// be
void pass_arguments(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use_result(hostrpc::page_t *page, uint64_t d[8])
{
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}
}  // namespace hostcall_ops
#endif

namespace hostrpc
{
template <typename Word, bool Inverted>
using bitmap_t = slot_bitmap<Word, __OPENCL_MEMORY_SCOPE_DEVICE, Inverted,
                             properties::fine_grain<Word>>;
}

template <typename Word>
static HOSTRPC_ANNOTATE Word bypass_load_t(HOSTRPC_ATOMIC(Word) * addr)
{
  return platform::atomic_load<Word, __ATOMIC_RELAXED,
                               __OPENCL_MEMORY_SCOPE_DEVICE>(addr);
}

MODULE(bitmap)
{
  using namespace hostrpc;
  using Word = uint64_t;

  enum : uint32_t
  {
    sz = 256
  };
  enum : uint32_t
  {
    words = sz / (8 * sizeof(Word))
  };

  auto active_threads = platform::all_threads_active_constant();

  static __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word)
      baseline_storage[words];
  static __attribute__((aligned(64))) HOSTRPC_ATOMIC(Word)
      inverted_storage[words];

  enum : Word
  {
    m1 = ~((Word)0)
  };
  auto bypass_load = bypass_load_t<Word>;

  for (uint32_t i = 0; i < words; i++)
    {
      baseline_storage[i] = 0;
      inverted_storage[i] = m1;
    }

  bitmap_t<Word, false> baseline(baseline_storage);
  bitmap_t<Word, true> inverted(inverted_storage);

  TEST("check initial words")
  {
    if (platform::is_master_lane(active_threads))
      {
        for (uint32_t i = 0; i < words; i++)
          {
            CHECK(baseline.load_word(sz, i) == 0);
            CHECK(inverted.load_word(sz, i) == 0);
            CHECK(bypass_load(&baseline_storage[i]) == 0);
            CHECK(bypass_load(&inverted_storage[i]) == m1);
          }
      }
  }

  TEST("check initial bits")
  {
    if (platform::is_master_lane(active_threads))
      {
        bool ok = true;
        for (uint32_t i = 0; i < sz; i++)
          {
            hostrpc::port_t p = static_cast<hostrpc::port_t>(i);
            ok &= baseline.read_bit(sz, p) == 0;
            ok &= inverted.read_bit(sz, p) == 0;
          }
        CHECK(ok);
      }
  }

  TEST("can set all bits")
  {
    if (platform::is_master_lane(active_threads))
      {
        bool ok = true;
        for (uint32_t i = 0; i < sz; i++)
          {
            hostrpc::port_t p = static_cast<hostrpc::port_t>(i);
            baseline.claim_slot(sz, p);
            inverted.claim_slot(sz, p);
          }
        for (uint32_t i = 0; i < words; i++)
          {
            ok &= (baseline.load_word(sz, i) == m1);
            ok &= (inverted.load_word(sz, i) == m1);
            ok &= (bypass_load(&baseline_storage[i]) == m1);
            ok &= (bypass_load(&inverted_storage[i]) == 0);
          }
        for (uint32_t i = 0; i < sz; i++)
          {
            hostrpc::port_t p = static_cast<hostrpc::port_t>(i);
            baseline.release_slot(sz, p);
            inverted.release_slot(sz, p);
          }
        for (uint32_t i = 0; i < words; i++)
          {
            ok &= (baseline.load_word(sz, i) == 0);
            ok &= (inverted.load_word(sz, i) == 0);
            ok &= (bypass_load(&baseline_storage[i]) == 0);
            ok &= (bypass_load(&inverted_storage[i]) == m1);
          }
        CHECK(ok);
      }
  }

  TEST("permute each slot in sequence")
  {
    if (platform::is_master_lane(active_threads))
      {
        bool ok = true;
        for (uint32_t i = 0; i < sz; i++)
          {
            hostrpc::port_t p = static_cast<hostrpc::port_t>(i);

            ok &= baseline.read_bit(sz, p) == 0;
            baseline.claim_slot(sz, p);
            ok &= baseline.read_bit(sz, p) == 1;
            baseline.release_slot(sz, p);
            ok &= baseline.read_bit(sz, p) == 0;
            baseline.toggle_slot(sz, p);
            ok &= baseline.read_bit(sz, p) == 1;
            baseline.toggle_slot(sz, p);
            ok &= baseline.read_bit(sz, p) == 0;

            ok &= inverted.read_bit(sz, p) == 0;
            inverted.claim_slot(sz, p);
            ok &= inverted.read_bit(sz, p) == 1;
            inverted.release_slot(sz, p);
            ok &= inverted.read_bit(sz, p) == 0;
            inverted.toggle_slot(sz, p);
            ok &= inverted.read_bit(sz, p) == 1;
            inverted.toggle_slot(sz, p);
            ok &= inverted.read_bit(sz, p) == 0;
          }
        CHECK(ok);
      }
  }

  TEST("check cas (uses exchange_weak everywhere at present)")
  {
    if (platform::is_master_lane(active_threads))
      {
        for (uint32_t i = 0; i < words; i++)
          {
            Word bl;
            CHECK(baseline.load_word(sz, i) == 0);
            while (!baseline.cas(i, 0, m1, &bl))
              ;
            CHECK(bl == 0);
            CHECK(baseline.load_word(sz, i) == m1);
            while (!baseline.cas(i, m1, 0, &bl))
              ;
            CHECK(bl == m1);
            CHECK(baseline.load_word(sz, i) == 0);

            Word il;
            CHECK(inverted.load_word(sz, i) == 0);
            while (!inverted.cas(i, 0, m1, &il))
              ;
            CHECK(il == 0);
            CHECK(inverted.load_word(sz, i) == m1);
            while (!inverted.cas(i, m1, 0, &il))
              ;
            CHECK(il == m1);
            CHECK(inverted.load_word(sz, i) == 0);
          }
      }
  }
}

MAIN_MODULE() { DEPENDS(bitmap); }
