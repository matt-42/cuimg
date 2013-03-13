#ifndef CUIMG_BOX2D_H_
# define CUIMG_BOX2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/border.h>

namespace cuimg
{
  class box2d
  {
  public:

    __host__ __device__ inline box2d();
    __host__ __device__ inline box2d(i_int2 a, i_int2 b);
    __host__ __device__ inline box2d(const box2d& d);

    __host__ __device__ inline box2d& operator=(const box2d& d);

    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;

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

  inline __host__ __device__
  bool operator==(const box2d& a, const box2d& b)
  {
    return a.nrows() == b.nrows() && a.ncols() == b.ncols();
  }

  inline __host__ __device__
  box2d operator-(const obox2d& d, const border& bd)
  {
    int b = bd.thickness();
    return box2d(i_int2(b, b),
                 i_int2(d.nrows() - b - 1, d.ncols() - b - 1));
  }


  inline __host__ __device__
  box2d operator-(const box2d& d, const border& bd)
  {
    int b = bd.thickness();
    return box2d(i_int2(d.p1().r() + b, d.p1().c() + b),
                 i_int2(d.p2().r() - b, d.p2().c() - b));
  }

}

# include <cuimg/box2d.hpp>

#endif
