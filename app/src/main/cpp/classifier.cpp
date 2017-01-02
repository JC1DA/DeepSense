#include <classifier.hpp>
#include <basic_functions.hpp>
#include <clio.hpp>
#include <malloc.h>
#include <stdio.h>
#include <deepsense_internal_lib.hpp>
#include <deepsense_lib.hpp>

float 	*	cnn_doClassification(cnn_frame *frame, cnn *model) {
    cnn_frame *result = frame;

    double totalTime = 0;
    double t0,t1;
    double global_t0 = get_timestamp();

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err;
    cl_event event;

    for(int i = 0 ; i < model->nLayers ; i++) {
        cnn_layer *layer = &model->layers[i];

        t0 = get_timestamp();

        result = layer->doFeedForward(result, layer);

        if(result->useGPU) {
            int size = result->w * result->h * result->c * (layer->useHalf ? sizeof(cl_half) : sizeof(cl_float));
            float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					result->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					size, \
					0, NULL, NULL, &err);
            SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

            if(!result->useHalf)
                LOGD("1st data: %f", buf_dest[0]);

            clEnqueueUnmapMemObject(openCLObjects->queue, \
					result->cl_data, \
					buf_dest, \
					0, NULL, NULL);
        }

        t1 = get_timestamp();
        double milsecs = (t1 - t0) / 1000.0L;

        LOGD("Processed layer %d in %f ms\n", (i + 1), milsecs);
    }

    if(result != NULL && result->useGPU) {

        result = frame_convert_to_gpu_float(result);

        result->data = (float *)malloc(result->w * result->h * result->c * sizeof(float));

        err = clEnqueueReadBuffer (openCLObjects->queue,
                                   result->cl_data,
                                   true,
                                   0,
                                   result->w * result->h * result->c * sizeof(float),
                                   result->data,
                                   0,
                                   0,
                                   0);

        err |= clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err  = clReleaseMemObject(result->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        result->useGPU = 0;
    }

    timestamp_t global_t1 = get_timestamp();

    totalTime = (global_t1 - global_t0) / 1000.0L;

    LOGD("CNN finished in %f ms\n", totalTime);

    float *output = (result == NULL) ? NULL : result->data;

    if(result != NULL)
        frame_free(result);

    return output;
}
