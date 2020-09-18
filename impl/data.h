// derived from aomp openmp runtime

#ifndef SRC_RUNTIME_INCLUDE_DATA_H_
#define SRC_RUNTIME_INCLUDE_DATA_H_

#include "hsa.h"
#include "hsa_ext_amd.h"

namespace core {


// TODO: Drop this
#define ATMI_WAIT_STATE HSA_WAIT_STATE_BLOCKED
  
inline hsa_status_t invoke_hsa_copy(hsa_signal_t sig, void *dest,
                                    const void *src, size_t size,
                                    hsa_agent_t agent) {
  const hsa_signal_value_t init = 1;
  const hsa_signal_value_t success = 0;
  hsa_signal_store_screlease(sig, init);

  hsa_status_t err =
      hsa_amd_memory_async_copy(dest, agent, src, agent, size, 0, NULL, sig);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // async_copy reports success by decrementing and failure by setting to < 0
  hsa_signal_value_t got = init;
  while (got == init) {
    got = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_NE, init,
                                    UINT64_MAX, ATMI_WAIT_STATE);
  }

  if (got != success) {
    return HSA_STATUS_ERROR;
  }

  return err;
}

} // namespace core
#endif // SRC_RUNTIME_INCLUDE_DATA_H_
