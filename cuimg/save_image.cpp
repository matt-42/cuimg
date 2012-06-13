# include <opencv/cv.h>
# include <opencv/highgui.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  void save_image(const std::string& filename, const host_image2d<i_uchar3>& in)
  {
    cv::Mat out(in.nrows(), in.ncols(), CV_8UC3);
    for (int i = 0; i < out.rows; i++)
      memcpy(out.ptr(i), &in(i, 0), out.cols * sizeof(i_uchar3));
    cv::imwrite(filename, out);
  }
}
