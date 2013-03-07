#ifndef CUIMG_KERNEL_TYPE_H_
# define CUIMG_KERNEL_TYPE_H_

namespace cuimg
{

  template <typename T>
  struct kernel_type
  {
    typedef T ret;
  };

}

#endif
