#ifndef CUIMG_NEIGHB_ITERATOR2D_H_
# define CUIMG_NEIGHB_ITERATOR2D_H_

# include <cuimg/point2d.h>

namespace cuimg
{
  template <typename N>
  class neighb_iterator2d
  {
  public:
    typedef point2d<int> point;

    __host__  __device__ inline neighb_iterator2d(const point& p, const N& n);


     __host__  __device__ inline bool is_valid() const;
     __host__  __device__ inline void invalidate();
     __host__  __device__ inline void next();
     __host__  __device__ inline void start();

     __host__  __device__ inline int i() const { return i_ - 1; }

     __host__  __device__ inline const point& operator*() const;
     __host__  __device__ inline const point* operator->() const;
     __host__  __device__ inline operator point() const { return p_; }

  private:
    const point cur_;
    const N n_;
    point p_;
    int i_;
  };
};

# include <cuimg/neighb_iterator2d.hpp>

# define for_all(p) for(p.start(); p.is_valid(); p.next())

#endif
