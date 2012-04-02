#ifndef CUIMG_STATIC_NEIGHB2D_H_
# define CUIMG_STATIC_NEIGHB2D_H_

# include <cuimg/point2d.h>

namespace cuimg
{
  template <typename P>
  struct add_ptr
  {
      typedef P* ret;
  };

  template <unsigned S>
  class static_neighb2d
  {
  public:
    typedef point2d<int> point;

    __host__  __device__ inline static_neighb2d(const int n[S][2])    : dpoints_(n)
  {
  }

    __host__  __device__ inline static_neighb2d(const static_neighb2d<S>& n);
    __host__  __device__ inline static_neighb2d<S>& operator=(const static_neighb2d<S>& n);

    __host__  __device__ inline unsigned size() const;
    __host__  __device__ inline const point& operator[](unsigned s) const;

  private:
    add_ptr<const int[2]>::ret dpoints_;
  };

}

# include <cuimg/static_neighb2d.hpp>

#endif
