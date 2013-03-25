#ifndef CUIMG_BOX2D_H_
# define CUIMG_BOX2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>

namespace cuimg
{
  class box2d
  {
  public:

    __host__ __device__ inline box2d();
    __host__ __device__ inline box2d(i_int2 a, i_int2 b);
    __host__ __device__ inline box2d(const box2d& d);

    __host__ __device__ inline box2d& operator=(const box2d& d);

    __host__ __device__ inline int nrows() const;
    __host__ __device__ inline int ncols() const;

    __host__ __device__ inline const i_int2& p1() const;
    __host__ __device__ inline const i_int2& p2() const;
    __host__ __device__ inline const i_int2 center() const;
    __host__ __device__ inline int size() const;

    __host__ __device__ inline bool has(const point2d<int>& p) const;

    __host__ __device__ inline void extend(const point2d<int>& p);
    __host__ __device__ inline void extend(const box2d& bb);

  private:
    i_int2 a_;
    i_int2 b_;
  };

}

# include <cuimg/box2d.hpp>

#endif
