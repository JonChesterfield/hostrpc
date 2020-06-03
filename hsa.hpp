#ifndef HSA_HPP_INCLUDED
#define HSA_HPP_INCLUDED

// A C++ wrapper around a subset of the hsa api
#include "hsa.h"
#include <cstdio>

namespace hsa
{

  const char*  status_string(hsa_status_t status)
  {
    const char * res;
    if (hsa_status_string(status, &res) 
        != HSA_STATUS_SUCCESS)
      {
        res = "unknown";
      }
    return res;
  }

  struct init
  {
    init() : status(hsa_init())
    {
      printf("called init: %u %s\n", status, status_string(status));
    }
    ~init()
    {
      hsa_shut_down();
    }
    const hsa_status_t status;
  };


}

#endif
