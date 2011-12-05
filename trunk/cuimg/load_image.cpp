# include <cv.h>
# include <highgui.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  void load_image(const std::string& filename, host_image2d<i_uchar3>& out)
  {
    cv::Mat m = cv::imread(filename, 1);
    out = host_image2d<i_uchar3>(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
      memcpy(&(out(i, 0)), m.ptr(i), m.cols * sizeof(i_uchar3));
  }
}
