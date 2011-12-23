# include <cv.h>
# include <highgui.h>

# include <cuimg/background_video_capture.h>


namespace cuimg
{

  void background_video_capture_loop(background_video_capture* v)
  {
    while (true) v->prepare_next_frame();
  }

  background_video_capture::background_video_capture(const background_video_capture& b)
  {
  }

  background_video_capture::background_video_capture()
    : cap_(),
      is_empty_cond_()
  {
  }

  background_video_capture::background_video_capture(const std::string& filename)
    : cap_(filename),
      is_empty_cond_()
  {
    if (is_opened()) 
    {
      init();
      start_producer_thread();
    }
  }

  background_video_capture::background_video_capture(int device)
    : cap_(device),
      is_empty_cond_()
  {
    if (is_opened()) 
    {
      init();
      start_producer_thread();
    }
  }

  void
  background_video_capture::init()
  {
    cap_.set(CV_CAP_PROP_CONVERT_RGB, true);
    buffer_.resize(max_buffer_size_);
    for (unsigned i = 0; i < buffer_.size(); i++)
      buffer_[i] = host_image2d<i_uchar3>(nrows(), ncols());
  }

  void background_video_capture::start_producer_thread()
  {
    reader_pos_ = -1;
    producer_pos_ = -1;
    producer_thread_ = boost::thread(background_video_capture_loop, this);
  }

  void
  background_video_capture::resize(unsigned nrows, unsigned ncols)
  {
    cap_.resize(nrows, ncols);
  }

  background_video_capture::~background_video_capture()
  {
  }

  bool background_video_capture::open(const std::string& filename)
  {
    bool b = cap_.open(filename);
    if (is_opened()) 
    {
      init();
      start_producer_thread();
    }
    return b;
  }

  bool background_video_capture::open(int device)
  {
    bool b = cap_.open(device);
    if (is_opened()) 
    {
      init();
      start_producer_thread();
    }
    return b;
  }

  bool background_video_capture::is_opened() const
  {
    return cap_.is_opened();
  }

  void background_video_capture::release()
  {
    return cap_.release();
  }

  bool background_video_capture::grab()
  {
    return cap_.grab();
  }

  bool background_video_capture::is_full() const
  {
    return producer_pos_ == (reader_pos_ + max_buffer_size_ - 1) % max_buffer_size_;
  }

  bool background_video_capture::is_empty() const
  {
    return producer_pos_ == reader_pos_;
  }

  host_image2d<i_uchar3>&
  background_video_capture::next_frame()
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (is_empty())
    {
      //std::cout << "waiting for producer..." << std::endl;
      is_empty_cond_.wait(lock);
    }
    reader_pos_ = (reader_pos_ + 1) % max_buffer_size_;
    //std::cout << "Consume: producer: " << producer_pos_ << "  consumer: " << reader_pos_ << std::endl;
    is_empty_cond_.notify_one();
    return buffer_[reader_pos_];
  }

  void
  background_video_capture::prepare_next_frame()
  {
    boost::unique_lock<boost::mutex> lock(mutex_);
    while (is_full())
    {
      //std::cout << "waiting for consumer..." << std::endl;
      is_empty_cond_.wait(lock);
    }
    int to_prepare = (producer_pos_ + 1) % max_buffer_size_;
    cap_ >> buffer_[to_prepare];
    producer_pos_ = to_prepare;
    //std::cout << "Produce: producer: " << producer_pos_ << "  consumer: " << reader_pos_ << std::endl;
    is_empty_cond_.notify_one();
  }

  unsigned background_video_capture::nrows()
  {
    return cap_.nrows();
  }

  void background_video_capture::rewind()
  {
    return cap_.rewind();
  }

  unsigned background_video_capture::ncols()
  {
    return cap_.ncols();
  }

  bool background_video_capture::finished()
  {
    return cap_.finished();
  }

  unsigned background_video_capture::nframes()
  {
    return cap_.nframes();
  }

  bool background_video_capture::set(int propId, double value)
  {
    return cap_.set(propId, value);
  }

  double background_video_capture::get(int propId)
  {
    return cap_.get(propId);
  }

}
