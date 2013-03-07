#include <cuimg/tracking2/bc2s_feature.h>

bool cuimg::cuda_bc2s_feature::cuda_bc2s_offsets_loaded_ = false;

__constant__ int cuimg::cuda_bc2s_offsets_s1[8];
__constant__ int cuimg::cuda_bc2s_offsets_s2[8];
