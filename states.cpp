#include "catch.hpp"
#include <cstdio>

struct host_machine
{
  bool G = 0;
  bool H = 0;
  bool T = 0;
};

struct gpu_machine
{
  bool G = 0;
  bool W = 0;
  bool H = 0;
};

struct slot
{
  host_machine host;
  gpu_machine gpu;

  void dump()
  {
    printf(
        "\n"
        "  G W H T\n"
        "H %u   %u %u\n"
        "D %u %u %u  \n",
        host.G, host.H, host.T, gpu.G, gpu.W, gpu.H);
  }
};

void invariants(slot s)
{
  // if gpu publishes work, it must have a wave waiting
  if (s.gpu.G)
    {
      assert(s.gpu.W);
    }
}

void host_read(slot &s) { s.host.G = s.gpu.G; }

void gpu_read(slot &s) { s.gpu.H = s.host.H; }

void host_release_slot(slot &s)
{
  assert(s.host.H);
  s.host.H = 0;
}

void gpu_release_slot(slot &s)
{
  assert(s.gpu.G);
  s.gpu.G = 0;
}

void wave_acquire_slot(slot &s)
{
  assert(s.gpu.W == 0);
  s.gpu.W = 1;
}

void wave_populate(slot &s)
{
  assert(s.gpu.W);
  assert(!s.gpu.G);
}

void wave_publish(slot &s)
{
  assert(s.gpu.W);
  assert(s.gpu.G == 0);
  s.gpu.G = 1;
}

void wave_receive(slot &s)
{
  assert(s.gpu.H);
  assert(s.gpu.W);
}

void wave_release_slot(slot &s)
{
  assert(s.gpu.W);
  s.gpu.W = 0;
}

void thread_acquire_slot(slot &s)
{
  assert(s.host.G);
  assert(s.host.T == 0);
  s.host.T = 1;
}

void thread_release_slot(slot &s)
{
  assert(s.host.H);
  assert(s.host.T);
  s.host.T = 0;
}

void thread_process(slot &s)
{
  assert(s.host.T);
  assert(!s.host.H);
}

void thread_publish(slot &s)
{
  assert(s.host.H == 0);
  s.host.H = 1;
}

TEST_CASE("happy path")
{
  slot s;

  wave_acquire_slot(s);
  wave_populate(s);
  wave_publish(s);

  host_read(s);
  thread_acquire_slot(s);
  thread_process(s);
  thread_publish(s);

  gpu_read(s);
  wave_receive(s);

  gpu_release_slot(s);

  host_read(s);
  thread_release_slot(s);
  host_release_slot(s);
  gpu_read(s);

  wave_release_slot(s);
}
