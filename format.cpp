#include <tuple>
#include <vector>

template <typename... Ts>
struct argStore
{
  using type = std::tuple<Ts...>;
  type value;

  argStore(Ts&&... args) : value(std::forward<Ts>(args)...) {}
};

enum argType
{
  argTypeInt,
  argTypeFloat,

};

template <typename... Ts>
std::vector<argType> encodeArgTypes();

// struct -> tuple?

template <typename... Ts>
int print(const char* fmt, Ts&&... ts)
{
  argStore<Ts...> args(std::forward<Ts>(ts)...);

  return 0;
}

int main() { print("%u", 42); }
