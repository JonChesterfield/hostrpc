server_t server;
client_t client;

int main() {
  const uint32_t calls = 10;
  auto box0 = make_unique<atomic_bool>();
  auto box1 = make_unique<atomic_bool>();
  auto data = make_unique<uint32_t[]>(4);

  client.inbox = server.outbox = box0.get();
  server.inbox = client.outbox = box1.get();
  client.buffer = server.buffer = data.get();

  thread st([]() -> void {
    for (uint32_t count = 0; count < 2 * calls;) {
      if (server.run(server_work, server_clean))
        count++;
    }
  });

  thread ct([]() -> void {
    for (uint32_t i = 0; i < calls; i++)
      client.run(client_fill, client_use);
  });

  st.join(); ct.join();
}
