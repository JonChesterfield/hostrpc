#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

POOL_INTERFACE_BOILERPLATE_HOST(example, 32);

uint32_t example::run(uint32_t state)
{
  if (platform::is_master_lane(platform::active_threads()))
    printf("run %u from %u (of %u/%u)\n", state, get_current_uuid(), alive(),
           requested());

  platform::sleep_briefly();
  return state + 1;
}

int main()
{
  example::initialize();  // may not need to do anything

  fprintf(stderr, "spawn\n");
  example::bootstrap_entry(8);

  // leave them running for a brief while
  usleep(10000);

  fprintf(stderr, "Start to wind down\n");

  example::teardown();

  example::finalize();

  return 0;
}
