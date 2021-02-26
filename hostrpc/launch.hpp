#ifndef LAUNCH_HPP_INCLUDED
#define LAUNCH_HPP_INCLUDED

#include "detail/platform_detect.hpp"

#if !HOSTRPC_HOST
#error "launch.hpp assumes host"
#endif

#include "hsa.hpp"

namespace
{
template <typename T>
struct with_implicit_args
{
  with_implicit_args(T s)
      : state(s), offset_x{0}, offset_y{0}, offset_z{0}, remainder{0}
  {
  }
  T state;
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  char remainder[80 - 24];
};

}  // namespace

template <typename T>
struct launch_t
{
  launch_t(const launch_t &) = delete;
  launch_t(launch_t &&o)
      : ready(o.ready),
        state(o.state),
        mutable_arg(o.mutable_arg),
        packet(o.packet)
  {
    o.ready = true;
    o.state = nullptr;
    o.mutable_arg = nullptr;
    o.packet = nullptr;
  }

  launch_t &operator=(const launch_t &) = delete;
  launch_t &operator=(launch_t &&o) = delete;

  launch_t()
      : ready(true), state(nullptr), mutable_arg(nullptr), packet(nullptr)
  {
  }

  hsa_kernel_dispatch_packet_t *setup_and_find_packet(hsa_agent_t kernel_agent,
                                                      hsa_queue_t *queue,
                                                      T args,
                                                      uint64_t *packet_id_ret)
  {
    // TODO: Clean this up
    ready = false;
    state = nullptr;
    mutable_arg = nullptr;
    packet = nullptr;

    hsa_region_t kernarg_region = hsa::region_kernarg(kernel_agent);
    hsa_region_t fine_grained_region = hsa::region_fine_grained(kernel_agent);
    // hsa_region_t coarse_grained_region =
    // hsa::region_coarse_grained(kernel_agent);

    // Copy args to fine grained memory
    auto mutable_arg_state = hsa::allocate(fine_grained_region, sizeof(T));
    if (!mutable_arg_state)
      {
        printf("Warning: allocate failed: %d\n", __LINE__);
        return nullptr;
      }

    mutable_arg = new (mutable_arg_state.get()) T(args);

    // Allocate kernarg memory, including implicit args
    void *kernarg_state =
        hsa::allocate(kernarg_region, sizeof(with_implicit_args<T *>))
            .release();
    if (!kernarg_state)
      {
        printf("Warning: allocate failed: %d\n", __LINE__);
        return nullptr;
      }

    state = new (reinterpret_cast<with_implicit_args<T *> *>(kernarg_state))
        with_implicit_args<T *>(mutable_arg);

    mutable_arg_state.release();

    uint64_t packet_id = hsa::acquire_available_packet_id(queue);

    *packet_id_ret = packet_id;
    const uint32_t mask = queue->size - 1;
    return (hsa_kernel_dispatch_packet_t *)queue->base_address +
           (packet_id & mask);
  }

  // takes ownership of signal
  launch_t(hsa_agent_t kernel_agent, hsa_queue_t *queue,
           hsa_signal_t &&completion, uint64_t kernel_address,
           uint32_t private_segment_fixed_size,
           uint32_t group_segment_fixed_size, uint64_t number_waves, T args)

  {
    assert(number_waves <= 8);
    uint64_t packet_id;
    packet = setup_and_find_packet(kernel_agent, queue, args, &packet_id);
    if (!packet)
      {
        return;
      }

    // This leaves private and group segment size at zero
    // That may be bad, could be the root cause of some crashes
    hsa::initialize_packet_defaults(packet);

    packet->kernel_object = kernel_address;
    memcpy(&packet->kernarg_address, &state, 8);

    packet->private_segment_size = private_segment_fixed_size;
    packet->group_segment_size = group_segment_fixed_size;

    packet->grid_size_x = packet->workgroup_size_x * number_waves;

    memcpy(&packet->completion_signal, &completion, sizeof(hsa_signal_t));

    hsa::    packet_store_release((uint32_t *)packet,
                                  hsa::                         header(HSA_PACKET_TYPE_KERNEL_DISPATCH),
                                  hsa::                         kernel_dispatch_setup());

    hsa_signal_store_release(queue->doorbell_signal, packet_id);
  }

  T operator()()
  {
    assert(state);
    wait();
    return *(state->state);
  }

  ~launch_t()
  {
    if (state)
      {
        wait();
        state->~with_implicit_args<T *>();
        hsa_memory_free(static_cast<void *>(state));
        hsa_memory_free(static_cast<void *>(mutable_arg));
        if (packet->completion_signal.handle)
          {
            hsa_signal_destroy(packet->completion_signal);
          }
        state = nullptr;
      }
  }

 private:
  void ring_doorbell() {}

  void wait()
  {
    if (ready)
      {
        return;
      }

    // completion signal may be null, in which case waiting on it doesn't work
    if (packet->completion_signal.handle != 0)
      {
        do
          {
          }
        while (hsa_signal_wait_acquire(
                   packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                   5000 /*000000*/, HSA_WAIT_STATE_ACTIVE) != 0);
      }
    ready = true;
  }

  bool ready;
  with_implicit_args<T *> *state;
  T *mutable_arg;
  hsa_kernel_dispatch_packet_t *packet;
};

template <typename T>
launch_t<T> launch(hsa_agent_t kernel_agent, hsa_queue_t *queue,
                   uint64_t kernel_address, T arg)
{
  return {kernel_agent, queue, kernel_address, arg};
}

#endif
