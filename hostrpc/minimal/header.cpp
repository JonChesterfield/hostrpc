#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

using namespace std;

struct process_t {
  const atomic_bool *inbox;
  atomic_bool *outbox;
  uint32_t *buffer;
};

struct client_t : public process_t {
  template <typename F, typename U> void run(F fill, U use);
};

struct server_t : public process_t {
  // return true if a callback was invoked
  template <typename W, typename C> bool run(W work, C clean);
};
