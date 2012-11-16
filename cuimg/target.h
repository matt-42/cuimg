#ifndef CUIMG_TARGET_H_
# define CUIMG_TARGET_H_

#ifndef __GNUC__
# define BOOST_TYPEOF_COMPLIANT
#endif
# include <boost/typeof/typeof.hpp>
# include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace cuimg
{
  enum target
  {
    CPU = 0,
    GPU = 1
  };

  template <cuimg::target F>
  struct flag
  {
  };

}

BOOST_TYPEOF_REGISTER_TEMPLATE(cuimg::flag, (BOOST_TYPEOF_INTEGRAL(cuimg::target)));

#endif // !CUIMG_TARGET_H_
