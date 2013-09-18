#ifndef CUIMG_GRID_H_
# define CUIMG_GRID_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>

namespace cuimg
{

  class grid;
  class grid_iterator : public std::iterator<std::input_iterator_tag, i_int2, i_int2>
  {
    unsigned short line_width_;
    i_int2 p_;

  public:
    inline __host__ __device__ grid_iterator(i_int2 p, grid b);
    inline __host__ __device__ grid_iterator(const grid_iterator& mit);
    inline __host__ __device__ grid_iterator& operator++();
    inline __host__ __device__ grid_iterator operator++(int);
    inline __host__ __device__ bool operator==(const grid_iterator& rhs);
    inline __host__ __device__ bool operator!=(const grid_iterator& rhs);
    inline __host__ __device__ const i_int2& operator*();
  };

  class grid
  {
  public:

    __host__ __device__ inline grid();
    __host__ __device__ inline grid(box2d a, int cellw, );
    __host__ __device__ inline grid(const grid& d);

    __host__ __device__ inline grid& operator=(const grid& d);

    __host__ __device__ inline int nrows() const;
    __host__ __device__ inline int ncols() const;

    __host__ __device__ inline const i_int2& p1() const;
    __host__ __device__ inline const i_int2& p2() const;
    __host__ __device__ inline const i_int2 center() const;
    __host__ __device__ inline int size() const;

    __host__ __device__ inline bool has(const point2d<int>& p) const;

    __host__ __device__ inline void extend(const point2d<int>& p);
    __host__ __device__ inline void extend(const grid& bb);

  private:
    i_int2 a_;
    i_int2 b_;
  };

}

# include <cuimg/grid.hpp>

#endif
