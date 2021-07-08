#include <math.h>
#include <float.h> // DBL_EPSILON

bool equals_double(double a, double b) {
  return fabs(a - b) <= DBL_EPSILON * fmax(1, fmax(fabs(a), fabs(b)));
}

bool equals_float(float a, float b) {
  return fabs(a - b) <= FLT_EPSILON * fmax(1, fmax(fabs(a), fabs(b)));
}
