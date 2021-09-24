#ifndef TIMER_HPP_INCLUDED
#define TIMER_HPP_INCLUDED

#include "platform/detect.hpp"

#if HOSTRPC_HOST
#include <chrono>
#include <iostream>
#include <string>

struct timer
{
  std::string n;
  std::chrono::high_resolution_clock::time_point tc;
  timer(std::string name)
      : n(name), tc(std::chrono::high_resolution_clock::now())
  {
  }
  ~timer()
  {
    try
      {
        std::chrono::high_resolution_clock::time_point td =
            std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(td - tc)
                .count();
        std::cout << n << ": " << duration << "ms\n";
      }
    catch (...)
      {
      }
  }
};

#endif
#endif
