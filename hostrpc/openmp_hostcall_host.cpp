struct operate {
  hsa_region_t coarse_region;
  operate(hsa_region_t r) : coarse_region(r) {}
  void op(hostrpc::cacheline_t *line);

  void operator()(hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      op(&page->cacheline[c]);
  }
};

struct clear {
  void operator()(hostrpc::page_t *page) {
    for (unsigned c = 0; c < 64; c++)
      page->cacheline[c].element[0] = opcodes_nop;
  }
};

// in a loop on a pthread,
// server->rpc_handle<operate, clear>(op, clear);

void operate::op(hostrpc::cacheline_t *line) {
  uint64_t op = line->element[0];
  switch (op) {
  case opcodes_nop: {
    break;
  }
  case opcodes_malloc: {
    uint64_t size;
    memcpy(&size, &line->element[1], 8);

    void *res;
    hsa_status_t r = hsa_memory_allocate(coarse_region, size, &res);
    if (r != HSA_STATUS_SUCCESS) {
      res = nullptr;
    }

    memcpy(&line->element[0], &res, 8);
    break;
  }
  case opcodes_free: {
    void *ptr;
    memcpy(&ptr, &line->element[1], 8);
    hsa_memory_free(ptr);
    break;
  }
  }
  return;
}
