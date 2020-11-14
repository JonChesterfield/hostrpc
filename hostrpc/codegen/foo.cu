#include "foo.hpp"

void unmarked_call()
{
  // treated as a __host__ function
  platform::foo();
}

__attribute__((host)) void host_call() { platform::foo(); }

__attribute__((device)) void device_call() { platform::foo(); }

__attribute__((host)) __attribute__((device)) void both_call()
{
  platform::foo();
}
