#ifndef HOSTRPC_TESTS_HPP_INCLUDED
#define HOSTRPC_TESTS_HPP_INCLUDED

#include <thread>

struct safe_thread
{
  template <typename Function, typename... Args>
  explicit safe_thread(Function f, Args... args)
      : t(std::forward<Function>(f), std::forward<Args>(args)...)
  {
  }
  ~safe_thread() { t.join(); }

 private:
  std::thread t;
};

#endif
