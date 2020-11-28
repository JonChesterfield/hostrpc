#ifndef HOSTRPC_X64_GCN_DEBUG_HPP_INCLUDE
#define HOSTRPC_X64_GCN_DEBUG_HPP_INCLUDE

#include <stdint.h>

namespace hostrpc
{
void debug(const char *);
void debug(const char *, uint64_t);
void debug(const char *, uint64_t, uint64_t);
void debug(const char *, uint64_t, uint64_t, uint64_t);
void debug(const char *, uint64_t, uint64_t, uint64_t, uint64_t);
}  // namespace hostrpc

#endif
