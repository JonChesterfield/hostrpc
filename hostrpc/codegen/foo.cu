#include "foo.hpp"

void unmarked_call(void)
{
  // treated as a __host__ function
  platform::foo();
}

HOSTRPC_HOST
void host_call(void) { platform::foo(); }

HOSTRPC_DEVICE
void device_call(void) { platform::foo(); }

HOSTRPC_HOST
HOSTRPC_DEVICE
void both_call(void) { platform::foo(); }
