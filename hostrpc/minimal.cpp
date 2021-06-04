#include <cstdint>
#include <cstdio>
#include <atomic>
#include <memory>
#include <thread>
#include <array>

struct process_t
{
  const std::atomic_bool * inbox;
  std::atomic_bool * outbox;
  uint32_t *buffer;
};


// TSAN doesn't do fences. It passes if the relaxed loads are tagged acquire
#define RELAXED_LOAD std::memory_order_relaxed
struct client_t : public process_t
{
  template <typename F, typename U>
  void run(F f, U u)
  {
    bool in = inbox->load(RELAXED_LOAD);
    bool out = outbox->load(RELAXED_LOAD);
    std::atomic_thread_fence(std::memory_order_acquire);

    if (!in & !out) {
      // ready! write to buffer then to outbox
      f(buffer);
      std::atomic_thread_fence(std::memory_order_release);
      outbox->store(1, std::memory_order_release);
      out = 1;
    }
   
    if (!in & out) {
      // wait for result
      while (!in) {in = inbox->load(RELAXED_LOAD); }
      std::atomic_thread_fence(std::memory_order_acquire);
    }

    if (in & out) {
      // use result
      std::atomic_thread_fence(std::memory_order_acquire);
      u(buffer);
      outbox->store(0, std::memory_order_release); out = 0;
    }

    if (in & !out) {
      /// wait for server to garbage collect
      while (in) {in = inbox->load(RELAXED_LOAD); }
      std::atomic_thread_fence(std::memory_order_acquire);
    }
  }
} client;


struct server_t : public process_t
{
  template <typename R, typename G>
  bool run(R r, G g)
  {
    bool in = inbox->load(RELAXED_LOAD);
    bool out = outbox->load(RELAXED_LOAD);
    std::atomic_thread_fence(std::memory_order_acquire);

    if (in & out) {
      // work already done
      while (in) {in = inbox->load(RELAXED_LOAD); }
      std::atomic_thread_fence(std::memory_order_acquire);
      return false;
    }

    if (!in & out)
      {
        g(buffer);
        std::atomic_thread_fence(std::memory_order_release);
        outbox->store(0, std::memory_order_release);
        out = 0;
        return true;
      }

    if (!in & !out)
      {
        while (!in) { in = inbox->load(RELAXED_LOAD); }
        std::atomic_thread_fence(std::memory_order_acquire);
        return false;
      }

    if (in & !out)
      {
        r(buffer);
        std::atomic_thread_fence(std::memory_order_release);
        outbox->store(1, std::memory_order_release);
        out = 1;
        return true;
      }

    return false;
  }
} server;



int main()
{
  auto box0 = std::make_unique<std::atomic_bool>();
  auto box1 = std::make_unique<std::atomic_bool>();
  auto data = std::make_unique<uint32_t[]>(4);

  client.inbox = server.outbox = box0.get();;
  server.inbox = client.outbox = box1.get();
  client.buffer = server.buffer = data.get();
  

  std::thread st([]()
                 {
                   for(uint32_t count = 0; count < 20;) {
                    bool ran  = server.run([](uint32_t * buffer)
                              {
                                for (int i = 0; i < 4; i++) {buffer[i]++; }
                              },
                     [](uint32_t * buffer)
                              {
                                for (int i = 0; i < 4; i++) {buffer[i] = 0; }
                              });
                   if (ran) count++;
                   }
                   return;
                 });


std::thread ct([]()
                 {
                   for (unsigned i = 0; i < 10; i++)
                     {
                   client.run([](uint32_t * buffer)
                              {
                                for (int i = 0; i < 4; i++) {buffer[i] = i; }
                              },
                     [](uint32_t * buffer)
                              {
                                printf("[");
                                for (int i = 0; i < 4; i++) {printf(" %u", buffer[i]); }
                                printf("]\n");
                              });
                     }
                   return;
                 });

 st.join();
 ct.join();
}
