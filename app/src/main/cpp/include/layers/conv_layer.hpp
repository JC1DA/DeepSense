#ifndef __CONV_LAYER__
#define __CONV_LAYER__

#include <deepsense_lib.hpp>

cnn_frame *doFeedForward_CONV(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_GPU(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_CONV_FC_GPU(cnn_frame *frame, void *layer);

#endif
