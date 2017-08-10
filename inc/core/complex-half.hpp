#ifndef COMPLEX_HALF_H
#define COMPLEX_HALF_H

#include <half-code/include/half.hpp>

#include <complex>

using float16 = half_float::half;

namespace gearshifft {

template<typename T> class complex;

  struct complex_float16 {
    float16 real;
    float16 imag;
  };

template<>
class complex<float16> {

public:
  typedef float16 value_type;
  typedef complex_float16 _ComplexT;


  constexpr complex(_ComplexT __z) : _M_value(__z) { }

  constexpr complex(float16 __r = float16(0), float16 __i = float16(0)) {
    _M_value.real = __r;
    _M_value.imag = __i;
  }

  constexpr float16
  real() const { return _M_value.real; }

  constexpr float16
  imag() const { return _M_value.imag; }


  template<typename T>
  void
  real(T __val) { _M_value.real = static_cast<float16>(__val); }

  template<typename T>
  void
  imag(T __val) { _M_value.imag = static_cast<float16>(__val); }

private:
  _ComplexT _M_value;
  };
}

#endif /* COMPLEX-HALF_H */
