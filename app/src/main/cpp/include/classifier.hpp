#ifndef __CLASSIFIER_HPP__
#define __CLASSIFIER_HPP__

#include "deepsense_lib.hpp"

float 			*	cnn_doClassification(cnn_frame *frame, cnn *model);
#endif
