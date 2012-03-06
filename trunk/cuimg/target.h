#ifndef CUIMG_TARGET_H_
# define CUIMG_TARGET_H_

namespace cuimg
{
  enum target
  {
    CPU = 3,
    GPU = 4
  };

  template <unsigned F>
  struct flag
  {
  };

}

#endif // !CUIMG_TARGET_H_
