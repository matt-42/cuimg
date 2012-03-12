# include <opencv/cv.h>
# include <opencv/highgui.h>

# include <cuimg/video_capture.h>


namespace cuimg
{

  video_capture::video_capture()
    : cap_(new cv::VideoCapture()),
      finished_(false)
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
    // cap_->set(CV_CAP_PROP_FORMAT, CV_8UC4);
  }

  video_capture::video_capture(const std::string& filename)
    : cap_(new cv::VideoCapture(filename)),
      finished_(false)
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
    // cap_->set(CV_CAP_PROP_FORMAT, CV_8UC4);
  }

  video_capture::video_capture(int device)
    : cap_(new cv::VideoCapture(device)),
      finished_(false)
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
    // cap_->set(CV_CAP_PROP_FORMAT, CV_8UC4);
  }

  void
  video_capture::resize(unsigned nrows, unsigned ncols)
  {
    cap_->set(CV_CAP_PROP_FRAME_WIDTH, ncols);
    cap_->set(CV_CAP_PROP_FRAME_HEIGHT, nrows);
  }

  video_capture::~video_capture()
  {
    delete cap_;
  }

  bool video_capture::open(const std::string& filename)
  {
    return cap_->open(filename);
  }

  bool video_capture::open(int device)
  {
    return cap_->open(device);
  }

  bool video_capture::is_opened() const
  {
    return cap_->isOpened();
  }

  void video_capture::release()
  {
    return cap_->release();
  }

  bool video_capture::grab()
  {
    return cap_->grab();
  }
/*
  bool video_capture::retrieve(Mat& image, int channel=0)
  {
    return cap_->retrieve(image, channel);
  }
*/

  video_capture&
  video_capture::operator>>(host_image2d<i_float1>& img)
  {
    // cap_->set(CV_CAP_PROP_FORMAT, CV_8UC4);
    get_frame((char*) img.data(), img.pitch(), sizeof(i_float1));
    return *this;
  }

  video_capture&
  video_capture::operator>>(host_image2d<i_uchar4>& img)
  {
    cap_->set(CV_CAP_PROP_FORMAT, CV_8UC4);
    get_frame((char*) img.data(), img.pitch(), sizeof(i_uchar4));
    return *this;
  }


  video_capture&
  video_capture::operator>>(host_image2d<i_uchar3>& img)
  {
     cap_->set(CV_CAP_PROP_FORMAT, CV_8UC3);

    // assert(img.data());
    // assert(cap_);
    // static cv::Mat m;
    // finished_ = !cap_->grab();
    // if (!cap_->retrieve(m))
    //   finished_ = true;
    // if (finished_)
    // {
    //   std::cout << "[Warning] video end... "  << std::endl;
    //   return *this;
    // }

    // std::cout << "copy... "  << std::endl;

    // for (int i = 0; i < m.rows; i++)
    // {
    //   assert(m.ptr(i));
    //   memcpy(img.data(), m.ptr(i), m.cols * sizeof(i_uchar3));
    // }

    get_frame((char*) img.data(), img.pitch(), sizeof(i_uchar3));
    return *this;
  }

  void
  video_capture::get_frame(char* buffer, unsigned pitch, unsigned pixel_sizeof)
  {
    assert(buffer);
    assert(cap_);
    static cv::Mat m;
    finished_ = !cap_->grab();
    if (!cap_->retrieve(m))
      finished_ = true;
    if (finished_)
      return;

    for (int i = 0; i < m.rows; i++)
    {
      assert(m.ptr(i));
      memcpy(buffer + i * pitch, m.ptr(i), m.cols * pixel_sizeof);
    }
  }

  unsigned video_capture::nrows()
  {
    return unsigned(cap_->get(CV_CAP_PROP_FRAME_HEIGHT));
  }

  void video_capture::rewind()
  {
    cap_->set(CV_CAP_PROP_POS_FRAMES, 0);
  }

  unsigned video_capture::ncols()
  {
    return unsigned(cap_->get(CV_CAP_PROP_FRAME_WIDTH));
  }

  bool video_capture::finished()
  {
    return finished_;//cap_->get(CV_CAP_PROP_POS_FRAMES) >= (cap_->get(CV_CAP_PROP_FRAME_COUNT) - 1);
  }

  unsigned video_capture::nframes()
  {
    return cap_->get(CV_CAP_PROP_POS_FRAMES);
  }

  bool video_capture::set(int propId, double value)
  {
    return cap_->set(propId, value);
  }

  double video_capture::get(int propId)
  {
    return cap_->get(propId);
  }

}
