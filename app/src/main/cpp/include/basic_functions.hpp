#ifndef __BASIC_FUNCTIONS_HPP__
#define __BASIC_FUNCTIONS_HPP__

#include "deepsense_lib.hpp"

int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4);
float getDataFrom4D(float *data, int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4);
int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3);
float getDataFrom3D(float *data, int d1, int d2, int d3, int i1, int i2, int i3);

cnn_frame *activate_RAMP(cnn_frame *frame);
cnn_frame *activate_LOGISTIC(cnn_frame *frame);
cnn_frame *activate_RELU(cnn_frame *frame);
cnn_frame *activate_LEAKY(cnn_frame *frame);
cnn_frame *doFeedForward_Activation(cnn_frame *frame, int activation);

cnn_frame		*	frame_init(int w, int h, int c);
cnn_frame 	    *	frame_init_gpu(int w, int h, int c);
cnn_frame       *   frame_init_gpu_half(int w, int h, int c);
cnn_frame       *   frame_clone(cnn_frame *src);
cnn_frame       *   frame_convert_to_gpu_float(cnn_frame *frame);
cnn_frame       *   frame_convert_to_gpu_half(cnn_frame *frame);
cnn_frame       *   frame_convert_to_cpu(cnn_frame *frame);
void				frame_free(cnn_frame *frame);

#endif
