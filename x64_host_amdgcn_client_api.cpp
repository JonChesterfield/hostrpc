#include "x64_host_amdgcn_client_api.hpp"
#include "base_types.hpp"
#include "interface.hpp"
#include "platform.hpp"

static const constexpr uint32_t MAX_NUM_DOORBELLS = 0x400;

namespace hostrpc
{
namespace x64_host_amdgcn_client_api
{
#if defined(__AMDGCN__)
void fill(hostrpc::page_t *page, void *dv)
{
  uint64_t *d = static_cast<uint64_t *>(dv);
  if (0)
    {
      // Will want to set inactive lanes to nop here, once there are some
      if (platform::is_master_lane())
        {
          for (unsigned i = 0; i < 64; i++)
            {
              page->cacheline[i].element[0] = 0;
            }
        }
    }

  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      line->element[i] = d[i];
    }
}
void use(hostrpc::page_t *page, void *dv)
{
  uint64_t *d = static_cast<uint64_t *>(dv);
  hostrpc::cacheline_t *line = &page->cacheline[platform::get_lane_id()];
  for (unsigned i = 0; i < 8; i++)
    {
      d[i] = line->element[i];
    }
}

#endif
#if defined(__x86_64__)
void operate(hostrpc::page_t *page, void *)
{
  for (unsigned c = 0; c < 64; c++)
    {
      hostrpc::cacheline_t &line = page->cacheline[c];
      for (unsigned i = 0; i < 8; i++)
        {
          line.element[i] = 2 * (line.element[i] + 1);
        }
    }
}
#endif
}  // namespace x64_host_amdgcn_client_api
}  // namespace hostrpc

using SZ = hostrpc::size_compiletime<hostrpc::x64_host_amdgcn_array_size>;

#if defined(__AMDGCN__)

__attribute__((visibility("default")))
hostrpc::x64_amdgcn_t::client_t client_singleton[MAX_NUM_DOORBELLS];

// Also in hsa.hpp
static uint16_t get_queue_index()
{
  static_assert(MAX_NUM_DOORBELLS < UINT16_MAX, "");
  uint32_t tmp0, tmp1;

  // Derived from mGetDoorbellId in amd_gpu_shaders.h, rocr
  // Using similar naming, exactly the same control flow.
  // This may be expensive enough to be worth caching or precomputing.
  uint32_t res;
  asm("s_mov_b32 %[tmp0], exec_lo\n\t"
      "s_mov_b32 %[tmp1], exec_hi\n\t"
      "s_mov_b32 exec_lo, 0x80000000\n\t"
      "s_sendmsg sendmsg(MSG_GET_DOORBELL)\n\t"
      "%=:\n\t"
      "s_nop 7\n\t"
      "s_bitcmp0_b32 exec_lo, 0x1F\n\t"
      "s_cbranch_scc0 %=b\n\t"
      "s_mov_b32 %[ret], exec_lo\n\t"
      "s_mov_b32 exec_lo, %[tmp0]\n\t"
      "s_mov_b32 exec_hi, %[tmp1]\n\t"
      : [ tmp0 ] "=&r"(tmp0), [ tmp1 ] "=&r"(tmp1), [ ret ] "=r"(res));

  res %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(res);
}

void hostcall_client(uint64_t data[8])
{
  hostrpc::x64_amdgcn_t::client_t &c = client_singleton[get_queue_index()];

  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = c.invoke(d, d);
    }
}

void hostcall_client_async(uint64_t data[8])
{
  hostrpc::x64_amdgcn_t::client_t &c = client_singleton[get_queue_index()];
  bool success = false;

  while (!success)
    {
      void *d = static_cast<void *>(&data[0]);
      success = c.invoke_async(d, d);
    }
}

#else

// Get the start of the array
const char *hostcall_client_symbol() { return "client_singleton"; }

hostrpc::x64_amdgcn_t::server_t server_singleton[MAX_NUM_DOORBELLS];

uint16_t queue_to_index(hsa_queue_t *q)
{
  char *sig = reinterpret_cast<char *>(q->doorbell_signal.handle);
  int64_t kind;
  __builtin_memcpy(&kind, sig, 8);
  // TODO: Work out if any hardware that works for openmp uses legacy doorbell
  assert(kind == -1);
  sig += 8;

  const uint64_t MAX_NUM_DOORBELLS = 0x400;

  uint64_t ptr;
  __builtin_memcpy(&ptr, sig, 8);
  ptr >>= 3;
  ptr %= MAX_NUM_DOORBELLS;

  return static_cast<uint16_t>(ptr);
}

hostrpc::x64_amdgcn_t *stored_pairs[MAX_NUM_DOORBELLS] = {0};

int hostcall_server_init(hsa_queue_t *queue, hsa_region_t fine,
                         hsa_region_t gpu_coarse, void *client_address)
{
  // Creates one of these heap types per call (roughly, per queue)

  uint16_t queue_id = queue_to_index(queue);
  assert(stored_pairs[queue_id] == 0);

  // TODO: Avoid this heap alloc
  hostrpc::x64_amdgcn_t *res =
      new hostrpc::x64_amdgcn_t(fine.handle, gpu_coarse.handle);
  if (!res)
    {
      return 1;
    }

  hostrpc::x64_amdgcn_t::client_t *clients =
      static_cast<hostrpc::x64_amdgcn_t::client_t *>(client_address);

  clients[queue_id] = res->client();

  server_singleton[queue_id] = res->server();

  stored_pairs[queue_id] = res;
  return 0;
}

void hostcall_server_dtor(hsa_queue_t *queue)
{
  uint16_t queue_id = queue_to_index(queue);
  assert(stored_pairs[queue_id] != 0);
  delete stored_pairs[queue_id];
}

bool hostcall_server_handle_one_packet(hsa_queue_t *queue)
{
  uint16_t queue_id = queue_to_index(queue);
  static thread_local uint64_t loc;  // this needs to be per-queue really
  return server_singleton[queue_id].handle(nullptr, &loc);
}

#endif
