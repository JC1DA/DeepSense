#ifndef __LRN_HPP__
#define __LRN_HPP__

#include <deepsense_lib.hpp>

cnn_frame *doFeedForward_LRN(cnn_frame *frame, void *layer);
cnn_frame *doFeedForward_LRN_GPU(cnn_frame *frame, void *layer);

#endif
