#ifndef CUIMG_TARGET_H_
# define CUIMG_TARGET_H_

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

#endif // !CUIMG_TARGET_H_
