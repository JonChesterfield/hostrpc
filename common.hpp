#ifndef HOSTRPC_COMMON_H_INCLUDED
#define HOSTRPC_COMMON_H_INCLUDED

#include <cstdint>
#include <cstdalign>

namespace hostrpc
{
  struct cacheline
  {
    uint64_t element[8];
  };
  
  struct page
  {
   cacheline line [64];

  };


}

#endif
