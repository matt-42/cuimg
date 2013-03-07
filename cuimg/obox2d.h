#ifndef CUIMG_OBOX2D_H_
# define CUIMG_OBOX2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{

  class obox2d;
  class obox2d_iterator : public std::iterator<std::input_iterator_tag, i_int2, i_int2>
  {
    unsigned short line_width_;
    i_int2 p_;

  public:
    inline __host__ __device__ obox2d_iterator(i_int2 p, obox2d b);
    inline __host__ __device__ obox2d_iterator(const obox2d_iterator& mit);
    inline __host__ __device__ obox2d_iterator& operator++();
    inline __host__ __device__ obox2d_iterator operator++(int);
    inline __host__ __device__ bool operator==(const obox2d_iterator& rhs);
    inline __host__ __device__ bool operator!=(const obox2d_iterator& rhs);
    inline __host__ __device__ const i_int2& operator*();
  };

  class obox2d
  {
  public:
    typedef obox2d_iterator iterator;

    __host__ __device__ inline obox2d();
    __host__ __device__ inline obox2d(unsigned nrows, unsigned ncols);
    __host__ __device__ inline obox2d(const obox2d& d);

    __host__ __device__ inline obox2d& operator=(const obox2d& d);

    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline unsigned size() const;

    __host__ __device__ inline bool has(const point2d<int>& p) const;

    __host__ __device__ inline point2d<int> mod(const point2d<int>& p) const;

    __host__ __device__ inline iterator begin() const  { return obox2d_iterator(i_int2(0, 0), *this); }
    __host__ __device__ inline iterator end() const { return obox2d_iterator(i_int2(nrows(), 0), *this); }

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
