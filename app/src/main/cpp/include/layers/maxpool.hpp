#ifndef __MAXPOOL_HPP__
#define __MAXPOOL_HPP__

#include <deepsense_lib.hpp>

cnn_frame *doFeedForward_MAXPOOL(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_MAXPOOL_GPU(cnn_frame *frame, void *layer);

#endif
