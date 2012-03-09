#ifndef CUIMG_POINT3D_H_
# define CUIMG_POINT3D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{
  template <typename C>
  class point3d
  {
  public:
    typedef C coord;

    __host__ __device__ point3d();
    __host__ __device__ point3d(C sli, C row, C col);
    __host__ __device__ point3d(const point3d<C>& d);


    __host__ __device__ point3d<C>& operator=(const point3d<C>& d);

    __host__ __device__ C sli() const;
    __host__ __device__ C row() const;
    __host__ __device__ C col() const;

    template <typename T>
    __host__ __device__ point3d(const improved_builtin<T, 3>& bt);

    __host__ __device__ operator typename make_bt<C, 3>::ret() const;

  private:
    C coords_[3];
  };

}

# include <cuimg/point3d.hpp>

#endif
