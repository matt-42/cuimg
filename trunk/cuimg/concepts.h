#ifndef CUIMG_CONCEPTS_H_
# define CUIMG_CONCEPTS_H_

namespace cuimg
{

  template <typename O>
  class Object
  {
  };

  template <typename I>
  class Image : public Object<I>
  {
  };

  template <typename I>
  class Image2d : public Image<I>
  {
  };

  template <typename O>
  const O& exact(const Object<O>& img)
  {
    return *static_cast<const O*>(&img);
  }

  template <typename O>
  O& exact(Object<O>& img)
  {
    return *static_cast<O*>(&img);
  }

}

#endif
