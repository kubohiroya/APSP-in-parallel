#include <float.h> // DBL_EPSILON

#ifndef APSP_IN_PARALLEL_EQUALS_HPP
#define APSP_IN_PARALLEL_EQUALS_HPP

bool equals_double(double a, double b);

bool equals_float(float a, float b);

template<typename Number> bool equals(Number a, Number b){
  return fabs(a - b) <= FLT_EPSILON * fmax(1, fmax(fabs(a), fabs(b)));
}

#endif //APSP_IN_PARALLEL_EQUALS_HPP
