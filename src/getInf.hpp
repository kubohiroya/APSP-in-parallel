#include "inf.hpp"

#ifndef GETINF_H
#define GETINF_H

template<typename Number> inline Number getInf();
template<> inline double getInf<double>(){
  return DBL_INF;
}
template<> inline float getInf<float>(){
  return FLT_INF;
}
template<> inline int getInf<int>(){
  return INT_INF;
}

#endif

extern "C" int get_infinity_int();
extern "C" float get_infinity_float();
extern "C" double get_infinity_double();
