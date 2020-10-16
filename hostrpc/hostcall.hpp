#ifndef HOSTCALL_HPP_INCLUDED
#define HOSTCALL_HPP_INCLUDED

#include "base_types.hpp"
#include "detail/platform_detect.h"

#include <stddef.h>
#include <stdint.h>

// gpu client api
#if defined(__AMDGCN__) || defined(__CUDACC__)
void hostcall_client(uint64_t data[8]);
void hostcall_client_async(uint64_t data[8]);
#endif

// Implementation api. This construct is a singleton.
namespace hostcall_ops
{
#if (HOSTRPC_HOST)
void operate(hostrpc::page_t *page);
void clear(hostrpc::page_t *page);
#endif
#if (HOSTRPC_AMDGCN || HOSTRPC_NVPTX)
void pass_arguments(hostrpc::page_t *page, uint64_t data[8]);
void use_result(hostrpc::page_t *page, uint64_t data[8]);
#endif
}  // namespace hostcall_ops

#endif
