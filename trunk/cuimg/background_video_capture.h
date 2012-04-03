#ifndef CUIMG_BACKGROUND_VIDEO_CAPTURE_H_
# define CUIMG_BACKGROUND_VIDEO_CAPTURE_H_

# include <vector>
# include <boost/thread.hpp>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/video_capture.h>


namespace cv
{
  // Fwd declaration.
  class VideoCapture;
}


namespace cuimg
{

  class background_video_capture
  {
  private:
    background_video_capture(const background_video_capture& b);

  public:
    background_video_capture();
    background_video_capture(const std::string& filename);
    background_video_capture(int device);

    ~background_video_capture();
    bool open(const std::string& filename);
    bool open(int device);
    bool is_opened() const;
    void release();
    void resize(unsigned nrows, unsigned ncols);

    host_image2d<i_uchar3>& next_frame();
    void prepare_next_frame();

    bool thread_end() const;
    bool grab();

    bool set(int propId, double value);
    double get(int propId);

    unsigned nrows();
    unsigned ncols();
    bool finished();
    void rewind();
    unsigned nframes();

    void wait_until_full();

  protected:
    void init();
    void start_producer_thread();

    bool is_empty() const;
    bool is_full() const;

    video_capture cap_;
    int producer_pos_;
    int reader_pos_;
    static const int max_buffer_size_ = 5;
    std::vector<host_image2d<i_uchar3> > buffer_;
    ::boost::condition_variable is_empty_cond_;
    ::boost::mutex mutex_;
    ::boost::thread producer_thread_;

    bool thread_end_;
  };

}

#endif
