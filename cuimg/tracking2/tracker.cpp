#define UNIT_ID CUIMG_TRACKER_CPP

#include <cuimg/tracking2/tracker.h>
#include <cuimg/tracking2/tracking_strategies.h>

using namespace cuimg;

template class tracker<tracking_strategies::bc2s_fast_gradient_cpu>;
template class tracker<tracking_strategies::bc2s_mdfl_gradient_cpu>;
//template class tracker<tracking_strategies::bc2s_dense_gradient_cpu>;
