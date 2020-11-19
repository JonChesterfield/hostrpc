#include "openmp_plugins.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <libgen.h>
#include <link.h>
#include <memory>

namespace hostrpc
{
namespace
{
struct call_free
{
  template <typename T>
  void operator()(T *ptr)
  {
    free(ptr);
  }
};

std::unique_ptr<char, call_free> plugin_path()
{
  std::unique_ptr<char, call_free> res;

  void *libomptarget = dlopen("libomptarget.so", RTLD_NOW);

  if (!libomptarget)
    {
      return res;
    }

  // undecided whether closing libomptarget is safer than leaving it open

  struct link_map *map;

  int rc = dlinfo(libomptarget, RTLD_DI_LINKMAP, &map);

  if (0 == rc)
    {
      if (map)
        {
          auto real = std::unique_ptr<char, call_free>(strdup(map->l_name));
          if (real)
            {
              char *dir = dirname(real.get());  // mutates real
              if (dir)
                {
                  fprintf(stderr, "%s vs %s vs %s\n", map->l_name, real.get(),
                          dir);
                  res = std::unique_ptr<char, call_free>(strdup(dir));
                }
            }
        }
    }

  dlclose(libomptarget);
  return res;
}

static bool find_plugin(const char *dir, const char *name)
{
  const char *fmt = "%s/%s";
  int size = snprintf(nullptr, 0, fmt, dir, name);
  if (size > 0)
    {
      size++;  // nul
      auto buffer = std::unique_ptr<char, call_free>((char *)malloc(size));
      int rc = snprintf(buffer.get(), size, fmt, dir, name);
      if (rc > 0)
        {
          fprintf(stderr, "Seek %s\n", buffer.get());
          void *r = dlopen(buffer.get(), RTLD_NOW | RTLD_NOLOAD);
          if (r != nullptr)
            {
              dlclose(r);
              return true;
            }
        }
    }
  return false;
}

plugins find_plugins_impl()
{
  plugins res;

  // Load the openmp target regions linked to this binary
#pragma omp target
  asm("");

  auto dir = plugin_path();
  if (dir)
    {
      fprintf(stderr, "path %s\n", dir.get());
      res.amdgcn = find_plugin(dir.get(), "libomptarget.rtl.amdgpu.so");
      res.nvptx = find_plugin(dir.get(), "libomptarget.rtl.cuda.so");
    }

  return res;
}
}  // namespace

struct plugins_cache
{
 private:
  plugins value;
  static plugins_cache &getInstance()
  {
    static plugins_cache instance;
    return instance;
  }
  plugins_cache() { value = find_plugins_impl(); }
  plugins_cache(plugins_cache const &) = delete;
  void operator=(plugins_cache const &) = delete;

 public:
  static plugins get() { return getInstance().value; }
};

plugins find_plugins() { return plugins_cache::get(); }
}  // namespace hostrpc
