#ifndef INCPRINTF_HPP
#define INCPRINTF_HPP

#include <cstdint>
#include <vector>

struct incr
{
  incr() {}

  void set_format(const char* fmt);

  std::vector<char> finalize();

  template <typename T>
  void piecewise_pass_element_T(T value);

  std::vector<char> format;
  size_t loc = 0;
  std::vector<char> output;
};

#endif
