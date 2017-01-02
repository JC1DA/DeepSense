#ifndef __FULLY_CONNECTED_HPP__
#define __FULLY_CONNECTED_HPP__

#include <deepsense_lib.hpp>

cnn_frame *doFeedForward_FULLY_CONNECTED(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_FULLY_CONNECTED_GPU(cnn_frame *frame, void *layer);

#endif
