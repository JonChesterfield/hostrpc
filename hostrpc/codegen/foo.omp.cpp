#include "foo.hpp"

void call() { platform::foo(); }

#pragma omp declare target
void target_call()
{
  //  platform::foo();
}
#pragma omp end declare target
