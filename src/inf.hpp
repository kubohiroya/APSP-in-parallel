
#define INT_INF std::numeric_limits<int>::max() / 2

#define FLT_INF std::numeric_limits<float>::max() / 2

#define DBL_INF std::numeric_limits<double>::max() / 2

extern "C" int getIntegerInfinity();
extern "C" float getFloatInfinity();
extern "C" double getDoubleInfinity();
