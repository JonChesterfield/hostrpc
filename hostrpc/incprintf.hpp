#ifndef INCPRINTF_HPP
#define INCPRINTF_HPP

#include <cstdint>
#include <vector>

// an incremental formatter
// can be passed a *printf format string,
// followed by N arguments, and will
// return a vector<char> containing the result
struct incr
{
  incr() {}

  void set_format(const char* fmt);
  bool have_format() const { return format.size() != 0; }

  std::vector<char> finalize();

  template <typename T>
  int __printf_pass_element_T(T value);

  std::vector<char> format;
  size_t loc = 0;
  std::vector<char> output;

  std::vector<char> accumulator;  // for buffering cstr that arrive in pieces

  template <size_t N>
  void append_cstr_section(const char* s)
  {
    accumulator.insert(accumulator.end(), s, s + N);
  }
};

#endif
