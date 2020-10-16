#ifndef HOSTCALL_HSA_HPP_INCLUDED
#define HOSTCALL_HSA_HPP_INCLUDED

#include <stddef.h>

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

// TODO: runtime
namespace hostrpc
{
static const constexpr size_t x64_host_amdgcn_array_size = 2048;
}

#endif
