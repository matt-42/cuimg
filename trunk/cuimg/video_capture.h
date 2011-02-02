#ifndef CUIMG_VIDEO_CAPTURE_H_
# define CUIMG_VIDEO_CAPTURE_H_

# include <cuimg/host_image2d.h>

namespace cv
{
  // Fwd declaration.
  class VideoCapture;
}


namespace cuimg
{

  class video_capture
  {
  public:
    video_capture();
    video_capture(const std::string& filename);
    video_capture(int device);

    ~video_capture();
    bool open(const std::string& filename);
    bool open(int device);
    bool is_opened() const;
    void release();

    bool grab();
//    bool retrieve(host_image2d<i_uchar3>& image, int channel=0);

    video_capture& operator >> (host_image2d<i_uchar3>& image);

    bool set(int propId, double value);
    double get(int propId);

    unsigned nrows();
    unsigned ncols();
    bool finished();

  protected:
    cv::VideoCapture* cap_;
  };

}

#endif
