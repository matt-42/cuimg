#ifndef CUIMG_TARGET_H_
# define CUIMG_TARGET_H_

# define BOOST_TYPEOF_COMPLIANT
# include <boost/typeof/typeof.hpp>
# include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace cuimg
{
  enum target
  {
    CPU = 0,
    GPU = 1
  };

  template <unsigned F>
  struct flag
  {
  };

}

BOOST_TYPEOF_REGISTER_TEMPLATE(cuimg::flag, (unsigned));

#endif // !CUIMG_TARGET_H_
