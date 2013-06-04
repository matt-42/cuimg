#ifndef CUIMG_OMP_APPLY_H_
# define CUIMG_OMP_APPLY_H_

# ifndef NO_CPP0X
# include <thread>
# endif

# include <cuimg/architectures.h>
# include <cuimg/obox2d.h>
# include <cuimg/box2d.h>

namespace cuimg
{


  /* struct thread_pool_pthreads */
  /* { */
  /*   template <typename F> */
  /*   thread_pool_pthreads(int n, void f(void*)) */
  /*   { */
  /*     for (int i = 0; i < n; i++) */
  /*     { */
  /* 	pthread_t tid = 0; */
  /* 	pthread_create(&tid, 0, f, &i); */
  /* 	threads_.push_back(i); */
  /*     } */
  /*   } */

  /*   void join() */
  /*   { */
  /*     for (pthread_t t : threads_) */
  /* 	pthread_join(t, 0); */
  /*   } */

  /*   std::vector<pthread_t> threads_; */
  /* }; */

  struct thread_pool
  {
    template <typename F>
    thread_pool(int n, F f)
    {
      for (int i = 0; i < n; i++)
      {
  	threads_.push_back(std::thread(f, i));
      }
    }

    void join()
    {
      for (std::thread& t : threads_)
  	t.join();
    }

    std::vector<std::thread> threads_;
  };

#ifdef NO_OPENMP
  template <typename F>
  inline void mt_apply2d(int elt_size, const obox2d& domain, const F& f_, cpu)
  {
    F& f = *const_cast<F*>(&f_);
    dim3 dimblock = ::cuimg::dimblock(cpu(), elt_size, domain);

    int nthreads = cpu::ncores();
    thread_pool pool(nthreads,
  		     [&](int id)
  		     {
  		       for (unsigned r = dimblock.y * id; r < domain.nrows(); r+=dimblock.y*nthreads)
  			 for (unsigned ri = 0; ri < dimblock.y && ri + r < domain.nrows(); ri++)
  			   for (unsigned c = 0; c < domain.ncols(); c++)
  			     f(i_int2(r, c));
  		     });
    pool.join();
  }


  template <typename F>
  inline void mt_apply2d(int elt_size, const box2d& domain, const F& f_, cpu)
  {
    F& f = *const_cast<F*>(&f_);
    dim3 dimblock = ::cuimg::dimblock(cpu(), elt_size, domain);

    int nthreads = cpu::ncores();
    thread_pool pool(nthreads,
  		     [&](int id)
  		     {
  		       for (unsigned r = domain.p1().r() + dimblock.y * id; r < domain.p2().r(); r+=dimblock.y*nthreads)
  			 for (unsigned ri = 0; ri < dimblock.y && ri + r < domain.p2().r(); ri++)
  			   for (int c = domain.p1().c(); c <= domain.p2().c(); c++)
  			     f(i_int2(r, c));
  		     });
    pool.join();
  }
#else

  template <typename ARCH, typename F>
  inline void mt_apply2d(int elt_size, const obox2d& domain, const F& f_, ARCH = cpu())
  {
    F& f = *const_cast<F*>(&f_);
    dim3 dimblock = ::cuimg::dimblock(ARCH(), elt_size, domain);
#pragma omp parallel for schedule(static, dimblock.y)
    for (unsigned r = 0; r < domain.nrows(); r++)
      for (unsigned c = 0; c < domain.ncols(); c++)
        f(i_int2(r, c));
  }


  template <typename ARCH, typename F>
  inline void mt_apply2d(int elt_size, const box2d& domain, const F& f_, ARCH = cpu())
  {
    F& f = *const_cast<F*>(&f_);
    dim3 dimblock = ::cuimg::dimblock(ARCH(), elt_size, domain);
#pragma omp parallel for schedule(static, dimblock.y)
    for (int r = domain.p1().r(); r <= domain.p2().r(); r++)
      for (int c = domain.p1().c(); c <= domain.p2().c(); c++)
        f(i_int2(r, c));
  }
#endif

  template <typename ARCH, typename F>
  inline void st_apply2d(int elt_size, const box2d& domain, const F& f, ARCH = cpu())
  {
    for (int r = domain.p1().r(); r <= domain.p2().r(); r++)
      for (int c = domain.p1().c(); c <= domain.p2().c(); c++)
        f(i_int2(r, c));
  }


  template <typename F, typename D>
  inline void mt_apply2d(const D& domain, const F& f)
  {
    mt_apply2d(1, domain, f, cpu());
  }

}

#endif
