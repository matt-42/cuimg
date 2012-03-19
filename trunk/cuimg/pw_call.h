#ifndef CUIMG_PW_CALL_H_
# define CUIMG_PW_CALL_H_

# include <omp.h>
# include <boost/type_traits/function_traits.hpp>
# include <cuimg/target.h>

namespace cuimg
{
  template <unsigned T>
  struct thread_info
  {
    dim3 blockDim;
    dim3 blockIdx;
    dim3 threadIdx;
  };








  template <void F(thread_info<CPU>)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock )
  {
//#pragma omp parallel shared(dimgrid, dimblock )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti );
    }
}
  }

#ifdef NVCC
  template <void F(thread_info<GPU>)> 
  __global__ void pw_call_kernel()
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti );
  }

  template <void F(thread_info<GPU>)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock )
  {
    pw_call_kernel< F><<<dimgrid, dimblock>>>();
  }

#endif

  template <typename A1, void F(thread_info<CPU>, A1)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1 );
    }
}
  }

#ifdef NVCC
  template <typename A1, void F(thread_info<GPU>, A1)> 
  __global__ void pw_call_kernel(A1 a1)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1 );
  }

  template <typename A1, void F(thread_info<GPU>, A1)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1 )
  {
    pw_call_kernel<A1,  F><<<dimgrid, dimblock>>>(a1);
  }

#endif

  template <typename A1, typename A2, void F(thread_info<CPU>, A1, A2)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, void F(thread_info<GPU>, A1, A2)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2 );
  }

  template <typename A1, typename A2, void F(thread_info<GPU>, A1, A2)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2 )
  {
    pw_call_kernel<A1, A2,  F><<<dimgrid, dimblock>>>(a1, a2);
  }

#endif

  template <typename A1, typename A2, typename A3, void F(thread_info<CPU>, A1, A2, A3)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, void F(thread_info<GPU>, A1, A2, A3)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3 );
  }

  template <typename A1, typename A2, typename A3, void F(thread_info<GPU>, A1, A2, A3)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3 )
  {
    pw_call_kernel<A1, A2, A3,  F><<<dimgrid, dimblock>>>(a1, a2, a3);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, void F(thread_info<CPU>, A1, A2, A3, A4)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, void F(thread_info<GPU>, A1, A2, A3, A4)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4 );
  }

  template <typename A1, typename A2, typename A3, typename A4, void F(thread_info<GPU>, A1, A2, A3, A4)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4 )
  {
    pw_call_kernel<A1, A2, A3, A4,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, void F(thread_info<CPU>, A1, A2, A3, A4, A5)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, void F(thread_info<GPU>, A1, A2, A3, A4, A5)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, void F(thread_info<GPU>, A1, A2, A3, A4, A5)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
  }

#endif

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, typename A14, void F(thread_info<CPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)> 
  void pw_call(const flag<CPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13, A14 a14 )
  {
//#pragma omp parallel shared(dimgrid, dimblock, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 )
{
#pragma omp parallel for schedule(static, 1) collapse(2)
    for(unsigned bz = 0; bz < dimgrid.z; bz++)
    for(unsigned by = 0; by < dimgrid.y; by++)
    for(unsigned bx = 0; bx < dimgrid.x; bx++)

    for(unsigned tz = 0; tz < dimblock.z; tz++)

    for(unsigned ty = 0; ty < dimblock.y; ty++)
    for(unsigned tx = 0; tx < dimblock.x; tx++)
    {
      thread_info<CPU> ti;
      ti.blockDim = dimblock;
      ti.blockIdx = dim3(bx, by, bz);
      ti.threadIdx = dim3(tx, ty, tz);
      F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 );
    }
}
  }

#ifdef NVCC
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, typename A14, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)> 
  __global__ void pw_call_kernel(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13, A14 a14)
  {
    thread_info<GPU> ti;
    ti.blockDim = blockDim;
    ti.blockIdx = blockIdx;
    ti.threadIdx = threadIdx;

    F(ti, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 );
  }

  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7, typename A8, typename A9, typename A10, typename A11, typename A12, typename A13, typename A14, void F(thread_info<GPU>, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)> 
  void pw_call(const flag<GPU>&, dim3 dimgrid, dim3 dimblock, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6, A7 a7, A8 a8, A9 a9, A10 a10, A11 a11, A12 a12, A13 a13, A14 a14 )
  {
    pw_call_kernel<A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14,  F><<<dimgrid, dimblock>>>(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
  }

#endif


} // end of namespace cuimg.

#endif // !CUIMG_PW_CALL_H_

