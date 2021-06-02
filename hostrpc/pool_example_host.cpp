#include "detail/platform_detect.hpp"
#include "pool_interface.hpp"

POOL_INTERFACE_BOILERPLATE_HOST(example, 32);

void example::run()
{
  if (platform::is_master_lane())
    printf("run from %u (of %u/%u)\n", get_current_uuid(), alive(),
           requested());

  platform::sleep_briefly();
}

int main()
{
  example::initialize();  // may not need to do anything

  fprintf(stderr, "spawn\n");
  example::bootstrap_entry(8);

  // leave them running for a while
  usleep(1000000);

  fprintf(stderr, "Start to wind down\n");

  example::teardown();

  example::finalize();

  return 0;
}
