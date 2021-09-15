#include "relacy/relacy_std.hpp"

#ifdef RL_RELACY_HPP
#define NS rl
#else
#define NS std
#include <atomic>
#include <pthread.h>
#endif

#include <stdint.h>
#include <stdio.h>

#define yield() rl::rl_sched_yield($)

#include "minimal/header.cpp"

#include "minimal/main.cpp"

#include "minimal/client.cpp"
#include "minimal/server.cpp"

void client_fill(uint32_t *buffer)
{
  for (int i = 0; i < 4; i++)
    {
      buffer[i] = i + 3;
    }
}

void client_use(uint32_t *buffer)
{
  return;
  printf("[");
  for (int i = 0; i < 4; i++)
    {
      printf(" %u", buffer[i]);
    }
  printf("]\n");
}

void server_work(uint32_t *buffer)
{
  for (int i = 0; i < 4; i++)
    {
      buffer[i] *= buffer[i % 3];
    }
}

void server_clean(uint32_t *buffer)
{
  for (int i = 0; i < 4; i++)
    {
      buffer[i] = 0;
    }
}
