#ifndef UNUSED_HPP
#define UNUSED_HPP

#include <boost/config.hpp>

namespace gearshifft
{
  template< typename... Ts >
  BOOST_FORCEINLINE
  BOOST_CXX14_CONSTEXPR
  void
  ignore_unused( Ts const& ... )
  {}

  template< typename... Ts >
  BOOST_FORCEINLINE
  BOOST_CXX14_CONSTEXPR
  void
  ignore_unused()
  {}

}

#endif /* UNUSED_HPP */
