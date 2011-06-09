# include <cv.h>
# include <highgui.h>

# include <cuimg/video_capture.h>


namespace cuimg
{

  video_capture::video_capture()
    : cap_(new cv::VideoCapture())
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
  }

  video_capture::video_capture(const std::string& filename)
    : cap_(new cv::VideoCapture(filename))
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
  }

  video_capture::video_capture(int device)
    : cap_(new cv::VideoCapture(device))
  {
    cap_->set(CV_CAP_PROP_CONVERT_RGB, true);
  }

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
  video_capture::operator>>(host_image2d<i_uchar3>& img)
  {
    static cv::Mat m;
    (*cap_) >> m;
    for (int i = 0; i < m.rows; i++)
      memcpy(&img(i, 0), m.ptr(i), m.cols * sizeof(i_uchar3));
    return *this;
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
    return cap_->get(CV_CAP_PROP_POS_FRAMES) >= (cap_->get(CV_CAP_PROP_FRAME_COUNT) - 1);
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
