#include "pool_interface.hpp"

struct example : public pool_interface::default_pool<example, 16>
{
  void run()
  {
    printf("run from %u\n", get_current_uuid());
    platform::sleep();
  }
};

extern "C" int main()
{
  example::set_requested(1);

  printf("Hit line %u\n", __LINE__);
  printf("Hit line %u\n", __LINE__);

  example::set_requested(3);
  printf("Hit line %u\n", __LINE__);

  example::spawn();
  printf("Hit line %u\n", __LINE__);

  //  example::loop();
  printf("Hit line %u\n", __LINE__);

  platform::sleep();

  example::set_requested(1);
  platform::sleep();
  example::set_requested(0);
  platform::sleep();

  return 0;
}
