#ifndef CUIMG_POINT2D_H_
# define CUIMG_POINT2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{
  template <typename C>
  class point2d : public improved_builtin<C, 2>
  {
  public:
    typedef C coord;
    typedef improved_builtin<C, 2> super;

    __host__ __device__ point2d();
    __host__ __device__ point2d(C row, C col);

    template <typename D>
    __host__ __device__ point2d(const improved_builtin<D, 2>& d);

    template <typename D>
    __host__ __device__ point2d<C>& operator=(const point2d<D>& d);
    template <typename D>
    __host__ __device__ point2d<C>& operator=(const improved_builtin<D, 2>& d);

    __host__ __device__ C row() const;
    __host__ __device__ C col() const;

    __host__ __device__ C& row();
    __host__ __device__ C& col();

  };

}

# include <cuimg/point2d.hpp>

#endif
