#ifndef HOSTRPC_H_INCLUDED
#define HOSTRPC_H_INCLUDED

#include <stdint.h>

#if 0
// Note to self:
// Can build queue-local state on the address of the doorbell
// See amd_gpu_agent.cpp, especially bindtraphandler
// Can use thread-local state on x64
#endif

// Reads from and writes to data
void hostrpc_sync(uint64_t data[8]);

// Reads from data, no return value
void hostrpc_noreturn(const uint64_t data[8]);

// Reads from data, returns a token
uint64_t hostrpc_async_send(const uint64_t data[8]);

// Writes to data and uses token iff returns true
bool hostrpc_async_read(uint64_t token, uint64_t data[8]);

// Blocks until available, reads result, frees token
void hostrpc_blocking_read(uint64_t token, uint64_t data[8]);

// Test if results are available without destroying token
bool hostrpc_available(uint64_t token);

// Free token
void hostrpc_cancel(uint64_t token);

// Blocks until available, reads result
void hostrpc_read_without_cancel(uint64_t token, uint64_t data[8]);

#endif
