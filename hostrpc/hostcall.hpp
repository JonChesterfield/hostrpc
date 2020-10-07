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
#include <memory>

// x64 host API

// awkward in that it is essentially a per-gpu singleton, which suggests an
// array indexed by device_id or similar. could require the caller to maintain
// that array.

class hostcall_impl;
class hostcall
{
 public:
  hostcall(hsa_agent_t kernel_agent);
  ~hostcall();
  bool valid() { return state_.get() != nullptr; }

  int enable_executable(hsa_executable_t);
  int enable_queue(hsa_queue_t *queue);
  int spawn_worker(hsa_queue_t *queue);

  hostcall(const hostcall &) = delete;
  hostcall(hostcall &&) = delete;

 private:
  std::unique_ptr<hostcall_impl> state_;
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
