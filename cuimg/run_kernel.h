#ifndef CUIMG_RUN_KERNEL_H_
# define CUIMG_RUN_KERNEL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/error.h>

namespace cuimg
{

#define RUN_KERNEL1D(O, M, ARGS, SIZE, ARCH) run_kernel1d<O, &O::M>(O ARGS, (SIZE),(ARCH));

  template <typename F, void (F::*m)(int)>
  void run_kernel1d(const F& f_, int size, const cpu&)
  {
    F& f = *const_cast<F*>(&f_);
#pragma omp parallel for schedule(static, 300)
    for (int i = 0; i < size; i++)
      (f.*m)(i);
  }

  template <typename F>
  void run_kernel1d_functor(const F& f_, int size, const cpu&)
  {
    F& f = *const_cast<F*>(&f_);
#pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < size; i++)
      f(i);
  }

  template <typename F>
  void run_kernel2d_functor(const F& f_, obox2d bb, const cpu&)
  {
    F& f = *const_cast<F*>(&f_);
    mt_apply2d(1, bb, f, cpu());
/* #pragma omp parallel for schedule(static, 2) */
/*     for (int r = 0; r < bb.nrows(); r++) */
/*       for (int c = 0; c < bb.ncols(); c++) */
/* 	f(i_int2(r, c)); */
  }


  template <typename F>
  void run_kernel2d_functor(const F& f_, box2d bb, const cpu&)
  {
    F& f = *const_cast<F*>(&f_);
    mt_apply2d(1, bb, f, cpu());

/* #pragma omp parallel for schedule(static, 2) */
/*     for (int r = bb.p1().r(); r <= bb.p2().r(); r++) */
/*       for (int c = bb.p1().c(); c <= bb.p2().c(); c++) */
/* 	f(i_int2(r, c)); */
  }

  template <typename F>
  void run_kernel2d(F& f, int size, const cpu&)
  {
#pragma omp parallel for schedule(static, 300)
    for (int i = 0; i < size; i++)
      f(i);
  }


#ifdef NVCC
  template <typename P, typename T, void (T::*m)(P)>
  inline __device__
  void member_call_wrapper(T& o, P i)
  {
    (o.*m)(i);
  }

  template <typename F>
  __global__ void
  run_kernel1d_functor_kernel(F f, int size)
  {
    int i = thread_pos1d();
    if (i < size)
      f(i);
  }

  template <typename F, void (&f)(F&, int)>
  __global__ void
  run_kernel1d_kernel(F o, int size)
  {
    int i = thread_pos1d();
    if (i < size)
      f(o, i);
  }

  template <typename F>
  __global__ void
  run_kernel2d_functor_kernel(F f, const obox2d domain)
  {
    i_int2 i = thread_pos2d();
    if (domain.has(i))
      f(i);
  }

  template <typename F, void (&f)(F&, i_int2)>
  __global__ void
  run_kernel2d_kernel(F o, const obox2d domain)
  {
    i_int2 i = thread_pos2d();
    if (domain.has(i))
      f(o, i);
  }

  template <typename F, void (F::*m)(int)>
  void run_kernel1d(const F& f_, int size, const cuda_gpu&)
  {
    if (size == 0) return;

    F& f = *const_cast<F*>(&f_);
    run_kernel1d_kernel<F, member_call_wrapper<int, F, m> ><<<cuda_gpu::dimgrid1d(size), cuda_gpu::dimblock1d()>>>(f, size);
    check_cuda_error();
  }

  template <typename F>
  void run_kernel1d_functor(const F& f_, int size, const cuda_gpu&)
  {
    if (size == 0) return;

    F& f = *const_cast<F*>(&f_);
    dim3 dg = cuda_gpu::dimgrid1d(size);
    dim3 db = cuda_gpu::dimblock1d();
    run_kernel1d_functor_kernel<F><<<dg, db>>>(f, size);
    check_cuda_error();
  }

  template <typename F, void (F::*m)(i_int2)>
  void run_kernel2d(const F& f_, const obox2d& domain, const cuda_gpu&)
  {
    if (domain.size() == 0) return;

    F& f = *const_cast<F*>(&f_);
    run_kernel2d_kernel<F, member_call_wrapper<i_int2, F, m> ><<<cuda_gpu::dimgrid2d(domain), cuda_gpu::dimblock2d()>>>(f, domain);
  }

  template <typename F>
  void run_kernel2d_functor(const F& f_, const obox2d& domain, const cuda_gpu&)
  {
    if (domain.size() == 0) return;

    F& f = *const_cast<F*>(&f_);
    dim3 dg = cuda_gpu::dimgrid2d(domain);
    dim3 db = cuda_gpu::dimblock2d();
    run_kernel2d_functor_kernel<F><<<dg, db>>>(f, domain);
    check_cuda_error();
  }

  template <typename F>
  __global__ void
  run_kernel2d_functor_kernel(F f, const box2d domain, i_int2 o)
  {
    i_int2 i = thread_pos2d() + o;
    if (domain.has(i))
      f(i);
  }

  template <typename F>
  void run_kernel2d_functor(const F& f_, const box2d& domain, const cuda_gpu&)
  {
    if (domain.size() == 0) return;

    F& f = *const_cast<F*>(&f_);
    dim3 dg = cuda_gpu::dimgrid2d(domain);
    dim3 db = cuda_gpu::dimblock2d();
    run_kernel2d_functor_kernel<F><<<dg, db>>>(f, domain, i_int2(domain.p1()));
    check_cuda_error();
  }

#endif

}

#endif
