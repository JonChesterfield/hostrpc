#ifndef HSA_HPP_INCLUDED
#define HSA_HPP_INCLUDED

// A C++ wrapper around a subset of the hsa api
#include "hsa.h"
#include <array>
#include <cstdio>

#include <type_traits>

namespace hsa
{
inline const char* status_string(hsa_status_t status)
{
  const char* res;
  if (hsa_status_string(status, &res) != HSA_STATUS_SUCCESS)
    {
      res = "unknown";
    }
  return res;
}

#define hsa_success_or_exit(status) \
  hsa::success_or_exit_impl(__LINE__, __FILE__, status)
inline void success_or_exit_impl(int line, const char* file,
                                 hsa_status_t status)
{
  if (status == HSA_STATUS_SUCCESS)
    {
      return;
    }
  printf("HSA Failure at %s:%d (%u,%s)\n", file, line, (unsigned)status,
         status_string(status));
  exit(1);
}

struct init
{
  init() : status(hsa_init()) { hsa_success_or_exit(status); }
  ~init() { hsa_shut_down(); }
  const hsa_status_t status;
};

#if __cplusplus >= 201703L
#define requires_invocable_r(...) \
  static_assert(std::is_invocable_r<__VA_ARGS__>::value, "")
#else
#define requires_invocable_r(...) (void)0
#endif

template <typename C>
void iterate_agents(C cb)
{
  requires_invocable_r(hsa_status_t, C, hsa_agent_t);

  auto L = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    C* unwrapped = static_cast<C*>(data);
    return (*unwrapped)(agent);
  };

  // res documented to fail if &cb == NULL or runtime not initialised
  // it also returns HSA_STATUS_INFO_BREAK if a traversal was stopped early
  hsa_status_t res = hsa_iterate_agents(L, static_cast<void*>(&cb));
  if (res == HSA_STATUS_INFO_BREAK)
    {
      return;
    }

  hsa_success_or_exit(res);
}

template <typename T, hsa_agent_info_t req>
struct agent_get_info
{
  static T call(hsa_agent_t agent)
  {
    T res;
    hsa_status_t r = hsa_agent_get_info(agent, req, static_cast<void*>(&res));
    (void)r;
    return res;
  }
};

template <hsa_agent_info_t req, typename e, size_t w>
struct agent_get_info<std::array<e, w>, req>
{
  using T = std::array<e, w>;
  static T call(hsa_agent_t agent)
  {
    T res;
    hsa_status_t r =
        hsa_agent_get_info(agent, req, static_cast<void*>(res.data()));
    (void)r;
    return res;
  }
};

inline std::array<char, 64> agent_get_info_name(hsa_agent_t agent)
{
  return agent_get_info<std::array<char, 64>, HSA_AGENT_INFO_NAME>::call(agent);
}

inline std::array<char, 64> agent_get_info_vendor_name(hsa_agent_t agent)
{
  return agent_get_info<std::array<char, 64>, HSA_AGENT_INFO_VENDOR_NAME>::call(
      agent);
}

inline hsa_agent_feature_t agent_get_info_feature(hsa_agent_t agent)
{
  return agent_get_info<hsa_agent_feature_t, HSA_AGENT_INFO_FEATURE>::call(
      agent);
}

inline uint32_t agent_get_info_queues_max(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUES_MAX>::call(agent);
}

inline uint32_t agent_get_info_queue_min_size(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUE_MIN_SIZE>::call(agent);
}

inline uint32_t agent_get_info_queue_max_size(hsa_agent_t agent)
{
  return agent_get_info<uint32_t, HSA_AGENT_INFO_QUEUE_MAX_SIZE>::call(agent);
}

inline hsa_queue_type32_t agent_get_info_queue_type(hsa_agent_t agent)
{
  return agent_get_info<hsa_queue_type32_t, HSA_AGENT_INFO_QUEUE_TYPE>::call(
      agent);
}

inline hsa_device_type_t agent_get_info_device(hsa_agent_t agent)
{
  return agent_get_info<hsa_device_type_t, HSA_AGENT_INFO_DEVICE>::call(agent);
}

inline std::array<uint8_t, 128> agent_get_info_extensions(hsa_agent_t agent)
{
  return agent_get_info<std::array<uint8_t, 128>,
                        HSA_AGENT_INFO_EXTENSIONS>::call(agent);
}

inline uint16_t agent_get_info_version_major(hsa_agent_t agent)
{
  return agent_get_info<uint16_t, HSA_AGENT_INFO_VERSION_MAJOR>::call(agent);
}

inline uint16_t agent_get_info_version_minor(hsa_agent_t agent)
{
  return agent_get_info<uint16_t, HSA_AGENT_INFO_VERSION_MINOR>::call(agent);
}

struct executable
{
  // hsa expects executable management to be quite dynamic
  // one can load multiple shared libraries, which can probably reference
  // symbols from each other. It supports 'executable_global_variable_define'
  // which names some previously allocated memory. Or readonly equivalent. This
  // wrapper is
  executable(hsa_agent_t agent) : agent(agent), valid(false)
  {
    hsa_profile_t profile =
        HSA_PROFILE_BASE;  // HIP uses full, vega claims 'base', unsure
    hsa_default_float_rounding_mode_t default_rounding_mode =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
    const char* options = 0;
    hsa_executable_t e;
    hsa_status_t rc =
        hsa_executable_create_alt(profile, default_rounding_mode, options, &e);

    if (rc == HSA_STATUS_SUCCESS)
      {
        state = e;
        valid = true;
      }
  }

  void load_from_file(hsa_file_t file)
  {
    hsa_code_object_reader_t reader;
    hsa_status_t rc0 = hsa_code_object_reader_create_from_file(file, &reader);

    // TODO: per agent or per system? Leaning towards per agent, may want
    // different code on different gpus.
    hsa_loaded_code_object_t code;
    hsa_status_t rc1 =
        hsa_executable_load_program_code_object(state, reader, NULL, &code);

    hsa_status_t rc2 = hsa_executable_freeze(state, NULL);

    uint32_t vres;
    hsa_status_t rc3 = hsa_executable_validate(state, &vres);

    // At this point, if the above all succeeded, can query the executable for
    // symbol info etc
  }

  ~executable()
  {
    if (valid)
      {
        // This fails if the executable is invalid
        hsa_executable_destroy(state);
      }
  }

 private:
  hsa_agent_t agent;
  bool valid;

  // Need a sentinel to indicate that the executable was not constructed
  // successfully Looking under the covers, hsa_executable_t is a pointer to
  // heap allocated memory A reinterpret_cast of nullptr to uint64_t may be the
  // right value here
  hsa_executable_t state{0};
};

}  // namespace hsa

#endif
