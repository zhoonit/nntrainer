#include <iostream>
#include <sstream>
#include <string>

/** base setups */
template <typename T> struct prop_tag { using type = typename T::prop_tag; };

template <typename T> class Property {

public:
  virtual ~Property() = default;

  const T &get() const { return value; }

  T &get() { return value; }

  void set(const T &v) {
    if (!is_valid(v)) {
      throw std::invalid_argument("argument is not valid");
    }
    value = v;
  }

  virtual bool is_valid(const T &v) { return true; }

private:
  T value;
};

/** predefined tags for general use */
struct int_prop_tag {};
struct vector_prop_tag {};
struct dimension_prop_tag {};
struct double_prop_tag {};
struct str_prop_tag {};

template <typename... Tags> struct tag_cast;

/// base case of the tag cast
template <typename Tag, typename... Others> struct tag_cast<Tag, Others...> {
  using type = Tag;
};

/// normal case of the tag cast
template <typename Tag, typename BaseTag, typename... Others>
struct tag_cast<Tag, BaseTag, Others...> {
  using type = std::conditional_t<std::is_base_of<BaseTag, Tag>::value, BaseTag,
                                  typename tag_cast<Tag, Others...>::type>;
};

template <typename Tag> struct str_converter {};

template <> struct str_converter<str_prop_tag> {
  static std::string to_string(const std::string &value) { return value; }

  static std::string from_string(const std::string &str) { return str + "STR"; }
};

/** converter definition */
template <> struct str_converter<int_prop_tag> {
  static std::string to_string(const int value) {
    return std::to_string(value);
  }

  static int from_string(const std::string &value) { return std::stoi(value); }
};

/** dispatcher */
template <typename T> std::string to_string(const T &property) {
  using tag_type = typename tag_cast<typename prop_tag<T>::type,
                                     vector_prop_tag, int_prop_tag>::type;
  return str_converter<tag_type>::to_string(property.get());
}

template <typename T> void from_string(const std::string &str, T &property) {
  using tag_type =
    typename tag_cast<typename prop_tag<T>::type, int_prop_tag>::type;
  property.set(str_converter<tag_type>::from_string(str));
}

/** custom specialization setups */
class Unit : public Property<int> {
public:
  static constexpr const char *key = "unit";
  using prop_tag = int_prop_tag;
};

class NumInput : public Property<std::string> {
public:
  static constexpr const char *key = "num_input";
  using prop_tag = str_prop_tag;
};
