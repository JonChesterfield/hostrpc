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
template <typename state_machine>
using bitmap_t = slot_bitmap<state_machine, __OPENCL_MEMORY_SCOPE_DEVICE>;
}

template <typename Word>
static HOSTRPC_ANNOTATE Word bypass_load_t(HOSTRPC_ATOMIC(Word) * addr)
{
  return platform::atomic_load<Word, __ATOMIC_RELAXED,
                               __OPENCL_MEMORY_SCOPE_DEVICE>(addr);
}

template <typename state_machine>
static HOSTRPC_ANNOTATE bool read_bit(hostrpc::bitmap_t<state_machine> &map,
                                      uint32_t size, uint32_t i)

{
  using Word = typename state_machine::Word;
  uint32_t w = hostrpc::index_to_element<Word>(i);
  Word d = map.load_word(size, w);
  uint32_t subindex = hostrpc::index_to_subindex<Word>(i);
  return hostrpc::bits::nthbitset(d, subindex);
}

MODULE(bitmap)
{
  using namespace hostrpc;
  struct state_machine
  {
    using Word = uint64_t;
  };
  using Word = state_machine::Word;

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

  bitmap_t<state_machine> baseline(baseline_storage);
  bitmap_t<state_machine> inverted(inverted_storage);

  TEST("check initial words")
  {
    if (platform::is_master_lane(active_threads))
      {
        for (uint32_t i = 0; i < words; i++)
          {
            CHECK(baseline.load_word(sz, i) == 0);
            CHECK(~inverted.load_word(sz, i) == 0);
            CHECK(bypass_load(&baseline_storage[i]) == 0);
            CHECK(bypass_load(&inverted_storage[i]) == m1);
          }
      }
  }

  TEST("can set all bits")
  {
    if (platform::is_master_lane(active_threads))
      {
        bool ok = true;
        for (uint32_t i = 0; i < sz; i++)
          {
            uint32_t p = static_cast<uint32_t>(i);
            baseline.claim_slot(sz, p);
          }
        for (uint32_t i = 0; i < words; i++)
          {
            ok &= (baseline.load_word(sz, i) == m1);
            ok &= (bypass_load(&baseline_storage[i]) == m1);
          }
        for (uint32_t i = 0; i < sz; i++)
          {
            uint32_t p = static_cast<uint32_t>(i);
            baseline.release_slot(sz, p);
          }
        for (uint32_t i = 0; i < words; i++)
          {
            ok &= (baseline.load_word(sz, i) == 0);
            ok &= (bypass_load(&baseline_storage[i]) == 0);
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
            uint32_t p = static_cast<uint32_t>(i);

            ok &= read_bit(baseline, sz, p) == 0;
            baseline.claim_slot(sz, p);
            ok &= read_bit(baseline, sz, p) == 1;
            baseline.release_slot(sz, p);
            ok &= read_bit(baseline, sz, p) == 0;
          }
        CHECK(ok);
      }
  }
}

MAIN_MODULE() { DEPENDS(bitmap); }
