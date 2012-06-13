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
    while (true)
    {
      boost::this_thread::interruption_point();
      o->prepare_next_frame();
    }
  }

  template <class M>
  global_mvt_thread<M>::global_mvt_thread(const domain_t& d)
  : matcher_(0),
    particles_(d.nrows() * d.ncols())

  {
    //cudaStreamCreate(&cuda_stream_);

    start_producer_thread();
  }


  template <class M>
  global_mvt_thread<M>::~global_mvt_thread()
  {
    terminate_thread();
  }

  template <class M>
  void
  global_mvt_thread<M>::terminate_thread()
  {
    producer_thread_.interrupt();
    producer_thread_.join();
  }

  template <class M>
  void
  global_mvt_thread<M>::update(const M& m, unsigned l)
  {
    // std::cout << "update_" << std::endl;
    assert(!matcher_);
    matcher_ = &m;
    level_ = l;
    is_empty_cond_.notify_one();

    boost::unique_lock<boost::mutex> lock(mutex_);
  }

  template <class M>
  void
  global_mvt_thread<M>::synchronize()
  {
    ::boost::unique_lock<boost::mutex> lock(mutex_);
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
  global_mvt_thread<M>::reset_mvt()
  {
    ::boost::unique_lock<boost::mutex> lock(mutex_);
    mvt_ = i_short2(0, 0);
  }

#ifndef NO_CUDA
  template <class M>
  void
  global_mvt_thread<M>::read_back_particles(thrust::host_vector<typename M::particle>& particles_,
					    const thrust::device_vector<typename M::particle>& d_particles_,
					    unsigned nparticles)
  {
    cudaMemcpy(thrust::raw_pointer_cast(&particles_[0]),
	       thrust::raw_pointer_cast(&d_particles_[0]),
	       nparticles * sizeof(typename M::particle),
	       cudaMemcpyDeviceToHost);
  }
#endif

  template <class M>
  void
  global_mvt_thread<M>::read_back_particles(std::vector<typename M::particle>& particles_,
					    const std::vector<typename M::particle>& d_particles_,
					    unsigned nparticles)
  {
    memcpy(&particles_[0],
	   &d_particles_[0],
           nparticles * sizeof(typename M::particle));
  }

  template <class M>
  void
  global_mvt_thread<M>::prepare_next_frame()
  {
    // std::cout << "prepare_next_frame wait" << std::endl;
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (!matcher_)
      is_empty_cond_.wait(lock);

    // std::cout << "prepare_next_frame process" << std::endl;
    read_back_particles(particles_, matcher_->compact_particles(), matcher_->n_particles());
    // copy_async(matcher_->matches(), matches_[level_], 0);
    // copy_async(matcher_->particles(), particles_[level_], 0);
    // cudaStreamSynchronize(cuda_stream_);
    //while (cudaStreamQuery(cuda_stream_) != cudaSuccess);
    mvt_ = mvt_detector_.estimate(particles_, matcher_->n_particles());

    matcher_ = 0;
    // std::cout << "matcher_ == 0" << std::endl;
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
    producer_thread_ = boost::thread(&global_mvt_thread_loop<global_mvt_thread<M> >, this);
  }

}

#endif
