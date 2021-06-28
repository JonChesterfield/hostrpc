server_t server;
client_t client;

const uint32_t calls = 3;

void *sv_call(void *) {
  for (uint32_t count = 0; count < 2 * calls;) {
    if (server.run(server_work, server_clean))
      count++;
  }
  return NULL;
}

void *ct_call(void *) {
  for (uint32_t i = 0; i < calls; i++) {
    client.run(client_fill, client_use);
  }

  return NULL;
}

struct race_test : rl::test_suite<race_test, 2> {
  uint32_t data[4];

  race_test() {}

  ~race_test() {}
  void before() {
    NS::atomic_bool *box0p = new NS::atomic_bool;
    NS::atomic_bool *box1p = new NS::atomic_bool;

    client.inbox = server.outbox = box0p;
    server.inbox = client.outbox = box1p;
    client.buffer = server.buffer = data;

    client.outbox->store(0, NS::memory_order_release);
    server.outbox->store(0, NS::memory_order_release);
  }

  void thread(unsigned thread_index) {
    if (0 == thread_index) {
      client.run(client_fill, client_use);

      if (0)
        for (uint32_t i = 0; i < calls; i++) {
          client.run(client_fill, client_use);
        }
    } else {
      if (server.run(server_work, server_clean)) {
        // progress
      } else {
        yield();
      }

      if (0)
        for (uint32_t count = 0; count < 2 * calls;) {
          if (server.run(server_work, server_clean)) {
            count++;
          }
        }
    }
  }

  void after() {
    delete client.outbox;
    delete server.outbox;
  }
};

extern "C" int main(void) {

  rl::test_params p;

  p.search_type = rl::sched_random;
  p.context_bound = 3;

  p.iteration_count = 10000;
  p.execution_depth_limit = 1000000;
  rl::simulate<race_test>(p);

  return 0;
}
