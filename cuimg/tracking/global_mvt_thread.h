#ifndef CUIMG_GLOBAL_MVT_THREAD_H_
# define CUIMG_GLOBAL_MVT_THREAD_H_

# include <boost/thread.hpp>

# include <vector>

# ifndef NO_CUDA
# include <thrust/host_vector.h>
# include <thrust/device_vector.h>
# endif

# include <cuimg/tracking/large_mvt_detector.h>

namespace cuimg
{

  template <class M>
  class global_mvt_thread
  {
  private:
    global_mvt_thread(const global_mvt_thread& b);

  public:
    typedef obox2d domain_t;

    global_mvt_thread(const domain_t& d);
    ~global_mvt_thread();

    void update(const M& matcher, unsigned l);

    void synchronize();

    i_short2 mvt();
    void reset_mvt();

    void prepare_next_frame();
    void terminate_thread();

    large_mvt_detector<i_float1> mvt_detector();

  protected:
    void start_producer_thread();

#ifndef NO_CUDA
    void read_back_particles(thrust::host_vector<typename M::particle>& particles_,
			     const thrust::device_vector<typename M::particle>& d_particles_,
			     unsigned nparticles);
#endif
    void read_back_particles(std::vector<typename M::particle>& particles_,
			     const std::vector<typename M::particle>& d_particles_,
			     unsigned nparticles);

    const M* matcher_;
    unsigned level_;
    i_short2 mvt_;
    static const int max_buffer_size_ = 10;
    std::vector<host_image2d<i_uchar3> > buffer_;

    ::boost::condition_variable is_empty_cond_;
    ::boost::mutex mutex_;
    ::boost::thread producer_thread_;

# ifndef NO_CUDA
    thrust::host_vector<typename M::particle> particles_;
# else
    std::vector<typename M::particle> particles_;
# endif

    large_mvt_detector<i_float1> mvt_detector_;

    cudaStream_t cuda_stream_;
  };

}

# include <cuimg/tracking/global_mvt_thread.hpp>

#endif
