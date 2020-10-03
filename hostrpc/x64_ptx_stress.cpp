
#include "x64_host_ptx_client.hpp"

using x64_ptx_type =
    hostrpc::x64_ptx_pair_T<hostrpc::size_runtime, hostrpc::indirect::fill,
                            hostrpc::indirect::use, hostrpc::indirect::operate,
                            hostrpc::indirect::clear>;

#if defined(__CUDACC__)

#else

#include "catch.hpp"

TEST_CASE("x64_ptx_stress")
{
  size_t N = 128;
  x64_ptx_type p(N);
}

#endif
