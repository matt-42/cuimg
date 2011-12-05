#ifndef CUIMG_UTIL_H_
# define CUIMG_UTIL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/error.h>

namespace cuimg
{

  inline int idivup(int a, int b)
  {
    return (a % b == 0) ? a / b : a / b + 1;
  }

  #define CUIMG_PI 3.14159265f
}

#endif
