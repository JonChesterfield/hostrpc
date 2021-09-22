#ifndef HOSTRPC_PRINTF_ENABLE_H_INCLUDED
#define HOSTRPC_PRINTF_ENABLE_H_INCLUDED

#include "detail/platform_detect.hpp"

#include <stdint.h>
#include <stddef.h>


#if (HOSTRPC_HOST)
#include "hsa.h"
#ifdef __cplusplus
extern "C"
{
#endif
  int hostrpc_print_enable_on_hsa_agent(hsa_executable_t ex,
                                        hsa_agent_t kernel_agent);
#ifdef __cplusplus
}
#endif
#endif

#endif
