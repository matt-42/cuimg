#ifndef CUIMG_POINT2D_H_
# define CUIMG_POINT2D_H_

# include <cuda_runtime.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{
  template <typename C>
  class point2d
  {
  public:
    typedef C coord;

    __host__ __device__ point2d();
    __host__ __device__ point2d(C row, C col);
    __host__ __device__ point2d(const point2d<C>& d);


    __host__ __device__ point2d<C>& operator=(const point2d<C>& d);

    __host__ __device__ C row() const;
    __host__ __device__ C col() const;

    __host__ __device__ C& row();
    __host__ __device__ C& col();

    template <typename T>
    __host__ __device__ point2d(const improved_builtin<T, 2>& bt);

    __host__ __device__ operator typename make_bt<C, 2>::ret() const;

  private:
    C coords_[2];
  };

  template <typename C, typename D>
  __host__ __device__ inline bool operator==(const point2d<C>& a, const point2d<D>& b);

}

# include <cuimg/point2d.hpp>

#endif
