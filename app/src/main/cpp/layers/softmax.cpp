#include <layers/softmax.hpp>
#include <math.h>
#include <clio.hpp>
#include <malloc.h>
#include <basic_functions.hpp>

cnn_frame *doFeedForward_SOFTMAX(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    double dsum = 0;
    int i;

    frame = frame_convert_to_cpu(frame);

    for(i = 0 ; i < frame->c ; i++) {
        dsum += exp((double)frame->data[i]);
    }

    for(i = 0 ; i < frame->c ; i++) {
        frame->data[i] = (float)(exp((double)frame->data[i]) / dsum);
    }

    return frame;
}
