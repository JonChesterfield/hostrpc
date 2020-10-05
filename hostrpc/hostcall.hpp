#ifndef HOSTCALL_HPP_INCLUDED
#define HOSTCALL_HPP_INCLUDED

#include "base_types.hpp"
#include <stddef.h>
#include <stdint.h>

// amdgcn client api
#if defined(__AMDGCN__)
void hostcall_client(uint64_t data[8]);
void hostcall_client_async(uint64_t data[8]);
#endif

#if defined(__x86_64__)
#include "hsa.h"

// x64 host API

// awkward in that it is essentially a per-gpu singleton, which suggests an
// array indexed by device_id or similar. could require the caller to maintain
// that array.

// involves an array in gpu memory
int hostcall_initialize(hsa_agent_t);
int hostcall_destroy(void);

// each executable has a pointer to that array, do the relocation manually
int hostcall_load_executable(hsa_agent_t, hsa_executable_t);
int hostcall_unload_executable(hsa_executable_t);

// allocates memory
int hostcall_enable_queue(hsa_agent_t, hsa_queue_t *);
int hostcall_disable_queue(hsa_agent_t, hsa_queue_t *);

// zero to close down
int hostcall_spawn_worker_thread(hsa_queue_t *);

const char *hostcall_client_symbol();

class hostcall
{
 public:
  hostcall(hsa_executable_t executable, hsa_agent_t kernel_agent);
  hostcall(void *client_symbol_address, hsa_agent_t kernel_agent);
  ~hostcall();
  bool valid();
  int enable_queue(hsa_agent_t kernel_agent, hsa_queue_t *queue);
  int spawn_worker(hsa_queue_t *queue);

  hostcall(const hostcall &) = delete;
  hostcall(hostcall &&) = delete;

 private:
  using state_t = hostrpc::storage<80, 8>;
  state_t state;
};
#endif

// Implementation api. This construct is a singleton.
namespace hostcall_ops
{
#if defined(__x86_64__)
void operate(hostrpc::page_t *page);
void clear(hostrpc::page_t *page);
#endif
#if defined __AMDGCN__
void pass_arguments(hostrpc::page_t *page, uint64_t data[8]);
void use_result(hostrpc::page_t *page, uint64_t data[8]);
#endif
}  // namespace hostcall_ops

// TODO: runtime
namespace hostrpc
{
static const constexpr size_t x64_host_amdgcn_array_size = 2048;
}

#endif
