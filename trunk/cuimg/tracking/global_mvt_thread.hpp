#ifndef CUIMG_GLOBAL_MVT_THREAD_HPP_
# define CUIMG_GLOBAL_MVT_THREAD_HPP_

# include <boost/thread.hpp>

# include <cuimg/tracking/global_mvt_thread.h>
# include <cuimg/copy.h>

namespace cuimg
{

  template <typename M>
  void global_mvt_thread_loop(M* o)
  {
    while (!o->thread_end()) o->prepare_next_frame();
  }

  template <class M>
  global_mvt_thread<M>::global_mvt_thread(const domain_t& d)
  : matcher_(0),
    particles_(d.nrows() * d.ncols())

  {
    // cudaStreamCreate(&cuda_stream_);
    // start_producer_thread();
  }


  template <class M>
  global_mvt_thread<M>::~global_mvt_thread()
  {
    thread_end_ = true;
    is_empty_cond_.notify_one();
    producer_thread_.join();
  }

  template <class M>
  void
  global_mvt_thread<M>::update(const M& m, unsigned l)
  {
    assert(!matcher_);
    matcher_ = &m;
    level_ = l;
    is_empty_cond_.notify_one();
  }

  template <class M>
  void
  global_mvt_thread<M>::synchronize()
  {
    ::boost::unique_lock<boost::mutex> lock(mutex_);
  }

  template <class M>
  bool
  global_mvt_thread<M>::thread_end() const
  {
    return thread_end_;
  }

  template <class M>
  i_short2
  global_mvt_thread<M>::mvt()
  {
    ::boost::unique_lock<boost::mutex> lock(mutex_);
    return mvt_;
  }

  template <class M>
  void
  global_mvt_thread<M>::prepare_next_frame()
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (!matcher_ && !thread_end())
    {
      is_empty_cond_.wait(lock);
    }

    if (!thread_end())
    {
#ifndef NO_CUDA
      cudaMemcpy(thrust::raw_pointer_cast(&particles_[0]),
                 thrust::raw_pointer_cast(&matcher_->compact_particles()[0]),
                 matcher_->n_particles() * sizeof(typename M::particle),
                 cudaMemcpyDeviceToHost);
      // copy_async(matcher_->matches(), matches_[level_], 0);
      // copy_async(matcher_->particles(), particles_[level_], 0);
      // cudaStreamSynchronize(cuda_stream_);
      //while (cudaStreamQuery(cuda_stream_) != cudaSuccess);
      mvt_ = mvt_detector_.estimate(particles_, matcher_->n_particles());
      //mvt_ = i_short2(0, 0);
#endif
    }

    matcher_ = 0;
    is_empty_cond_.notify_one();
  }

  template <class M>
  large_mvt_detector<i_float1>
  global_mvt_thread<M>::mvt_detector()
  {
    return mvt_detector_;
  }


  template <class M>
  void global_mvt_thread<M>::start_producer_thread()
  {
    thread_end_ = false;
    producer_thread_ = boost::thread(&global_mvt_thread_loop<global_mvt_thread<M> >, this);
  }

}

#endif
