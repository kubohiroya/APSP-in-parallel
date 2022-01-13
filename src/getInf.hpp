#pragma once

#include "inf.hpp"

template<typename Number> Number getInf();
template<> double getInf<double>();
template<> float getInf<float>();
template<> int getInf<int>();

extern "C" int get_infinity_int();
extern "C" float get_infinity_float();
extern "C" double get_infinity_double();
