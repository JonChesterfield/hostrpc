void client_fill(uint32_t *);
void server_work(uint32_t *);
void client_use(uint32_t *);
void server_clean(uint32_t *);

server_t server;
client_t client;

int main() {

  const uint32_t calls = 10;
  auto box0 = std::make_unique<std::atomic_bool>();
  auto box1 = std::make_unique<std::atomic_bool>();
  auto data = std::make_unique<uint32_t[]>(4);

  client.inbox = server.outbox = box0.get();
  server.inbox = client.outbox = box1.get();
  client.buffer = server.buffer = data.get();

  std::thread st([]() -> void {
    for (uint32_t count = 0; count < 2 * calls;) {
      if (server.run(server_work, server_clean)) {
        count++;
      }
    }
  });

  std::thread ct([]() -> void {
    for (uint32_t i = 0; i < calls; i++) {
      client.run(client_fill, client_use);
    }
  });

  st.join();
  ct.join();
}

