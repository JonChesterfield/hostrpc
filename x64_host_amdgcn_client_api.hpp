#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>

// needs to scale with CUs
namespace hostrpc
{
static const constexpr size_t x64_host_amdgcn_array_size = 2048;
}

#if defined(__AMDGCN__)

#include <stdint.h>
void hostcall_client(uint64_t data[8]);
void hostcall_client_async(uint64_t data[8]);

#endif

#if defined(__x86_64__)

#include "hsa.h"

const char *hostcall_client_symbol();

class hostcall
{
 public:
  hostcall(hsa_executable_t executable, hsa_agent_t kernel_agent);
  bool valid();
  int enable_queue(hsa_queue_t *queue);
  int spawn_worker(hsa_queue_t *queue);

 private:
  using state_t = hostrpc::storage<128, 8>;
  state_t state;
};

#endif

// x64 uses inlined function pointers to provide a cleaner interface
// That's not working on amdgcn with clang-10 or tot at present.

namespace hostrpc
{
namespace x64_host_amdgcn_client_api
{
#if defined(__AMDGCN__)
void fill(hostrpc::page_t *, void *);
void use(hostrpc::page_t *, void *);

#endif
#if defined(__x86_64__)
void operate(hostrpc::page_t *, void *);
#endif
}  // namespace x64_host_amdgcn_client_api
}  // namespace hostrpc

#endif
