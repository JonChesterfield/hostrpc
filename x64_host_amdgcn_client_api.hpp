#ifndef HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED
#define HOSTRPC_X64_HOST_AMDGCN_CLIENT_API_HPP_INCLUDED

#if defined(__AMDGCN__)

// todo: wire up a host alternative?
#include <stdint.h>
void hostcall_client(uint64_t data[8]);
void hostcall_client_async(uint64_t data[8]);

#endif

#if defined(__x86_64__)

#include "hsa.h"

const char *hostcall_client_symbol();
void *hostcall_server_init(hsa_region_t fine, hsa_region_t gpu_coarse,
                           void *client_address);
void hostcall_server_dtor(void *);
bool hostcall_server_handle_one_packet(void *);  // return true for did work

#endif

#endif