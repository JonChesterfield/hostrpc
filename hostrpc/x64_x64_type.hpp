#ifndef HOSTRPC_X64_X64_TYPE_HPP_INCLUDED
#define HOSTRPC_X64_X64_TYPE_HPP_INCLUDED

#include "client_server_pair.hpp"

namespace hostrpc
{
template <typename SZ>
using x64_x64_type = client_server_pair_t<SZ, arch::x64, arch::x64>;

}  // namespace hostrpc

#endif
