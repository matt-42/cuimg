#ifndef CUIMG_OBOX2D_H_
# define CUIMG_OBOX2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  class obox2d
  {
  public:

    __host__ __device__ inline obox2d();
    __host__ __device__ inline obox2d(unsigned nrows, unsigned ncols);
    __host__ __device__ inline obox2d(const obox2d& d);

    __host__ __device__ inline obox2d& operator=(const obox2d& d);

    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;

    __host__ __device__ inline bool has(const point& p) const;

  private:
    unsigned short nrows_;
    unsigned short ncols_;
  };

  inline bool operator==(const obox2d& a, const obox2d& b)
  {
    return a.nrows() == b.nrows() && a.ncols() == b.ncols();
  }

  inline obox2d operator/(const obox2d& a, const float x)
  {
    return obox2d(a.nrows() / x, a.ncols() / x);
  }

}

# include <cuimg/obox2d.hpp>

#endif
