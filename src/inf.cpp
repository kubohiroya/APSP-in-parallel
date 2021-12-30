#include "inf.hpp"

int get_infinity_int(){
  return INT_INF;
}
float get_infinity_float(){
  return FLT_INF;
}
double get_infinity_double(){
  return DBL_INF;
}

template<> double getInf<double>(){
  return DBL_INF;
}
template<> float getInf<float>(){
  return FLT_INF;
}
template<> int getInf<int>(){
  return INT_INF;
}
