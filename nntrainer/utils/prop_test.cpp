#include "properties.h"

#include <tuple>
#include <vector>

class Visitor {

public:
  template <typename... Ts> void save_result(std::tuple<Ts...> &tup, int hi) {
    if (hi == 0) {
      auto callable = [this](auto &&prop, size_t index) {
        std::string key = std::remove_reference_t<decltype(prop)>::key;
        stored_result.emplace_back(key, to_string(prop));
      };
      iterate_prop(callable, tup);
    }
  }

  std::vector<std::pair<std::string, std::string>> &get_result(int hi) {
    return stored_result;
  }

private:
  template <size_t I = 0, typename Callable, typename... Ts>
  typename std::enable_if<I == sizeof...(Ts), void>::type
  iterate_prop(Callable c, std::tuple<Ts...> &tup) {
    // end of recursion;
  }

  template <size_t I = 0, typename Callable, typename... Ts>
  typename std::enable_if<(I < sizeof...(Ts)), void>::type
  iterate_prop(Callable c, std::tuple<Ts...> &tup) {
    c(std::get<I>(tup), I);

    iterate_prop<I + 1>(c, tup);
  }

  std::vector<std::pair<std::string, std::string>> stored_result;
};

int main() {
  auto tuple = std::make_tuple<Unit, NumInput>(Unit(), NumInput());

  Visitor v;

  std::vector<std::string> strs = {"12", "HI"};
  from_string("12", std::get<0>(tuple));
  from_string("HI", std::get<1>(tuple));

  v.save_result(tuple, 0);
  auto ret = v.get_result(0);

  for (auto &item : ret) {
    std::cout << item.first << '=' << item.second << std::endl;
  }

  // std::cout << std::get<0>(tuple).get() << std::endl;
  // std::cout << std::get<1>(tuple).get() << std::endl;
}
