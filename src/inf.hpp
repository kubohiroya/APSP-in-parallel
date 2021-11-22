#include <stdexcept>
#include <limits>

#define INT_INF std::numeric_limits<int>::max() / 2

#define FLT_INF std::numeric_limits<float>::max() / 2

#define DBL_INF std::numeric_limits<double>::max() / 2

extern "C" int get_infinity_int();
extern "C" float get_infinity_float();
extern "C" double get_infinity_double();
