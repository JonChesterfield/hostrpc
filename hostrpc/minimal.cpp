#include "minimal/header.cpp"

#include "minimal/client.cpp"
#include "minimal/server.cpp"

#include "minimal/main.cpp"

void client_fill(uint32_t *buffer) {
  for (int i = 0; i < 4; i++) {
    buffer[i] = i + 3;
  }
}

void client_use(uint32_t *buffer) {
  printf("[");
  for (int i = 0; i < 4; i++) {
    printf(" %u", buffer[i]);
  }
  printf("]\n");
}

void server_work(uint32_t *buffer) {
  for (int i = 0; i < 4; i++) {
    buffer[i] *= buffer[i % 3];
  }
}

void server_clean(uint32_t *buffer) {
  for (int i = 0; i < 4; i++) {
    buffer[i] = 0;
  }
}
