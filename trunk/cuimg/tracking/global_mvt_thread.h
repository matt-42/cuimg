#ifndef CUIMG_GLOBAL_MVT_THREAD_H_
# define CUIMG_GLOBAL_MVT_THREAD_H_

# include <boost/thread.hpp>
# include <thrust/host_vector.h>

# include <cuimg/tracking/large_mvt_detector.h>

namespace cuimg
{

  template <class M>
  class global_mvt_thread
  {
  private:
    global_mvt_thread(const global_mvt_thread& b);

  public:
    typedef obox2d<point2d<int> > domain_t;

    global_mvt_thread(const domain_t& d);
    ~global_mvt_thread();

    void update(const M& matcher, unsigned l);

    void synchronize();

    i_short2 mvt();

    bool thread_end() const;

    void prepare_next_frame();

    large_mvt_detector<i_float1> mvt_detector();

  protected:
    void start_producer_thread();


    const M* matcher_;
    unsigned level_;
    i_short2 mvt_;
    static const int max_buffer_size_ = 10;
    std::vector<host_image2d<i_uchar3> > buffer_;

    ::boost::condition_variable is_empty_cond_;
    ::boost::mutex mutex_;
    ::boost::thread producer_thread_;

    thrust::host_vector<typename M::particle> particles_;

    large_mvt_detector<i_float1> mvt_detector_;

    bool thread_end_;

    cudaStream_t cuda_stream_;
  };

}

# include <cuimg/tracking/global_mvt_thread.hpp>

#endif
