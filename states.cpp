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

  void dump();
};

void dump(slot s)
{
  printf(
      "  G W H T\n"
      "H %u   %u %u\n"
      "D %u %u %u  \n",
      s.host.G, s.host.H, s.host.T, s.gpu.G, s.gpu.W, s.gpu.H);
}

void dump(slot f, slot t)
{
  printf(
      "  G W H T          G W H T\n"
      "H %u   %u %u   >    H %u   %u %u\n"
      "D %u %u %u          D %u %u %u  \n",
      f.host.G, f.host.H, f.host.T, t.host.G, t.host.H, t.host.T, f.gpu.G,
      f.gpu.W, f.gpu.H, t.gpu.G, t.gpu.W, t.gpu.H);
}

void slot::dump() { ::dump(*this); }

void invariants(slot s)
{
  // if gpu publishes work, it must have a wave waiting
  if (s.gpu.G)
    {
      assert(s.gpu.W);
    }
}

enum class operation
{
  host_error,
  gpu_error,
  host_begin,
  gpu_begin,
  host_end,
  gpu_end,

  host_wait_for_G1,
  host_wait_for_G0,
  gpu_wait_for_H1,
  gpu_wait_for_H0,

  host_read,
  gpu_read,

  host_release_slot,
  gpu_release_slot,

  try_wave_acquire_slot,
  wave_populate,
  wave_publish,
  wave_receive,
  wave_release_slot,

  try_thread_acquire_slot,
  thread_process,
  thread_publish,
  thread_release_slot,
};

const char *str(operation op)
{
  switch (op)
    {
      default:
        return "unknown";
      case operation::host_error:
        return "host_error";
      case operation::gpu_error:
        return "gpu_error";
      case operation::host_begin:
        return "host_begin";
      case operation::gpu_begin:
        return "gpu_begin";
      case operation::host_end:
        return "host_end";
      case operation::gpu_end:
        return "gpu_end";
      case operation::host_wait_for_G1:
        return "host_wait_for_G1";
      case operation::host_wait_for_G0:
        return "host_wait_for_G0";

      case operation::gpu_wait_for_H1:
        return "gpu_wait_for_H1";
      case operation::gpu_wait_for_H0:
        return "gpu_wait_for_H0";

      case operation::host_read:
        return "host_read";
      case operation::gpu_read:
        return "gpu_read";
      case operation::host_release_slot:
        return "host_release_slot";
      case operation::gpu_release_slot:
        return "gpu_release_slot";
      case operation::try_wave_acquire_slot:
        return "try_wave_acquire_slot";
      case operation::wave_populate:
        return "wave_populate";
      case operation::wave_publish:
        return "wave_publish";
      case operation::wave_receive:
        return "wave_receive";
      case operation::wave_release_slot:
        return "wave_release_slot";
      case operation::try_thread_acquire_slot:
        return "try_thread_acquire_slot";
      case operation::thread_process:
        return "thread_process";
      case operation::thread_publish:
        return "thread_publish";
      case operation::thread_release_slot:
        return "thread_release_slot";
    }
}

void on_error(slot &) {}

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

bool try_wave_acquire_slot(slot &s)
{
  if (s.gpu.W == 0)
    {
      s.gpu.W = 1;
      return true;
    }
  else
    {
      return false;
    }
}

void wave_populate(slot &s)
{
  assert(s.host.H == 0);
  assert(s.gpu.W);
  assert(!s.gpu.G);
}

void wave_publish(slot &s)
{
  assert(s.host.H == 0);
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

bool try_thread_acquire_slot(slot &s)
{
  assert(s.host.G);
  if (s.host.T == 0)
    {
      s.host.T = 1;
      return true;
    }
  else
    {
      return false;
    }
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

void thread_release_slot(slot &s)
{
  assert(s.host.H);
  assert(s.host.T);
  s.host.T = 0;
}

static const bool verbose = true;

struct host_sm
{
  operation next = operation::host_begin;
  slot &s;
  uint64_t packet_limit = 1;
  uint64_t packet_count = 0;

  host_sm(slot &s) : s(s) {}

  bool step()
  {
    operation current_op = next;
    slot current_slot = s;

    switch (next)
      {
        default:
          {
            next = operation::host_error;
            break;
          }
        case operation::host_begin:
          {
            next = operation::host_wait_for_G1;
            break;
          }
        case operation::host_end:
          {
            if (packet_count < packet_limit)
              {
                next = operation::host_begin;
              }
            else
              {
                next = operation::host_end;
              }
            break;
          }

        case operation::host_wait_for_G1:
          {
            host_read(s);
            if (s.host.G)
              {
                next = operation::try_thread_acquire_slot;
              }
            else
              {
                next = operation::host_wait_for_G1;
              }

            break;
          }

        case operation::try_thread_acquire_slot:
          {
            bool got = try_thread_acquire_slot(s);
            if (got)
              {
                next = operation::thread_process;
              }
            else
              {
                next = operation::host_begin;
              }
            break;
          }
        case operation::thread_process:
          {
            thread_process(s);
            next = operation::thread_publish;
            break;
          }
        case operation::thread_publish:
          {
            thread_publish(s);
            next = operation::thread_release_slot;
            break;
          }
        case operation::thread_release_slot:
          {
            thread_release_slot(s);
            next = operation::host_wait_for_G0;
            break;
          }
        case operation::host_wait_for_G0:
          {
            host_read(s);
            if (s.host.G)
              {
                next = operation::host_wait_for_G0;
              }
            else
              {
                next = operation::host_release_slot;
              }
            break;
          }
        case operation::host_release_slot:
          {
            host_release_slot(s);
            packet_count++;
            next = operation::host_end;
            break;
          }
      }

    if (verbose)
      {
        printf("%s => %s\n", str(current_op), str(next));
        dump(current_slot, s);
        printf("\n");
      }

    return current_op != next;
  }
};

struct gpu_sm
{
  operation next = operation::gpu_begin;
  slot &s;

  uint64_t packet_limit = 1;
  uint64_t packet_count = 0;

  gpu_sm(slot &s) : s(s) {}

  bool step()
  {
    operation current_op = next;
    slot current_slot = s;

    switch (next)
      {
        default:
          {
            next = operation::gpu_error;
            break;
          }
        case operation::gpu_begin:
          {
            next = operation::try_wave_acquire_slot;
            break;
          }
        case operation::gpu_end:
          {
            if (packet_count < packet_limit)
              {
                next = operation::gpu_begin;
              }
            else
              {
                next = operation::gpu_end;
              }
            break;
          }

        case operation::try_wave_acquire_slot:
          {
            if (try_wave_acquire_slot(s))
              {
                next = operation::wave_populate;
              }
            else
              {
                next = operation::try_wave_acquire_slot;
              }
            break;
          }
        case operation::wave_populate:
          {
            wave_populate(s);
            next = operation::wave_publish;
            break;
          }
        case operation::wave_publish:
          {
            wave_publish(s);
            next = operation::gpu_wait_for_H1;
            break;
          }

        case operation::gpu_wait_for_H1:
          {
            gpu_read(s);
            if (s.gpu.H)
              {
                next = operation::wave_receive;
              }
            else
              {
                next = operation::gpu_wait_for_H1;
              }
            break;
          }
        case operation::wave_receive:
          {
            wave_receive(s);
            next = operation::gpu_release_slot;
            break;
          }
        case operation::gpu_release_slot:
          {
            gpu_release_slot(s);
            next = operation::gpu_wait_for_H0;
            break;
          }

        case operation::gpu_wait_for_H0:
          {
            gpu_read(s);
            if (s.gpu.H)
              {
                next = operation::gpu_wait_for_H0;
              }
            else
              {
                next = operation::wave_release_slot;
              }
            break;
          }

        case operation::wave_release_slot:
          {
            wave_release_slot(s);

            packet_count++;
            next = operation::gpu_end;
            break;
          }
      }

    if (verbose)
      {
        printf("%s => %s\n", str(current_op), str(next));
        dump(current_slot, s);
        printf("\n");
      }

    return current_op != next;
  }
};

TEST_CASE("happy path")
{
  slot s;

  try_wave_acquire_slot(s);
  wave_populate(s);
  wave_publish(s);

  host_read(s);
  try_thread_acquire_slot(s);
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

TEST_CASE("Run one packet sequentially, host first")
{
  slot s;
  host_sm h(s);
  gpu_sm g(s);

  CHECK(h.next == operation::host_begin);
  CHECK(g.next == operation::gpu_begin);

  while (h.step())
    ;
  CHECK(h.next == operation::host_wait_for_G1);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_wait_for_H1);

  while (h.step())
    ;
  CHECK(h.next == operation::host_wait_for_G0);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_wait_for_H0);

  while (h.step())
    ;
  CHECK(h.next == operation::host_end);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_end);
}

TEST_CASE("Run one packet sequentially, gpu first")
{
  slot s;
  host_sm h(s);
  gpu_sm g(s);

  CHECK(h.next == operation::host_begin);
  CHECK(g.next == operation::gpu_begin);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_wait_for_H1);

  while (h.step())
    ;
  CHECK(h.next == operation::host_wait_for_G0);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_wait_for_H0);

  while (h.step())
    ;
  CHECK(h.next == operation::host_end);

  while (g.step())
    ;
  CHECK(g.next == operation::gpu_end);
}

TEST_CASE("interleave, host first")
{
  slot s;
  host_sm h(s);
  gpu_sm g(s);

  bool progress = true;
  while (progress)
    {
      progress = false;
      progress |= h.step();
      progress |= g.step();
    }

  CHECK(h.next == operation::host_end);
  CHECK(g.next == operation::gpu_end);
}

TEST_CASE("interleave, gpu first")
{
  slot s;
  host_sm h(s);
  gpu_sm g(s);

  bool progress = true;
  while (progress)
    {
      progress = false;
      progress |= g.step();
      progress |= h.step();
    }

  CHECK(h.next == operation::host_end);
  CHECK(g.next == operation::gpu_end);
}
