#include <layers/fully_connected.hpp>
#include <basic_functions.hpp>
#include <predefine.hpp>
#include <clio.hpp>
#include <deepsense_internal_lib.hpp>

cnn_frame *doFeedForward_FULLY_CONNECTED(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = frame_convert_to_cpu(frame);

    cnn_layer_fully_connected *connected_layer = ((cnn_layer *)layer)->connected_layer;

    cnn_frame *output = frame_init(1, 1, connected_layer->outputSize);

    for(int n = 0 ; n < connected_layer->outputSize ; n++) {
        output->data[n] = 0;
        for(int i = 0 ; i < frame->c ; i++) {
            for(int j = 0 ; j < frame->h ; j++) {
                for(int k = 0 ; k < frame->w ; k++) {
                    int index = getIndexFrom3D(frame->c, frame->h, frame->w, i , j , k);
                    output->data[n] += frame->data[index] * connected_layer->W[index * connected_layer->outputSize + n];
                }
            }
        }

        output->data[n] += connected_layer->bias[n];
    }

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_FULLY_CONNECTED_GPU(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cl_int err;
    cnn_layer_fully_connected *connected_layer = ((cnn_layer *)layer)->connected_layer;
    OpenCLObjects *openCLObjects = getOpenClObject();

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(1, 1, connected_layer->outputSize) : frame_init_gpu(1, 1, connected_layer->outputSize);

    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->fully_connected_kernel.kernel : openCLObjects->fully_connected_kernel_float.kernel;
    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &frame->cl_data);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &connected_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &connected_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &output->cl_data);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &connected_layer->outputSize);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    //size_t globalSize[1] = {(size_t)connected_layer->outputSize};
    size_t globalSize[1] = {(size_t)256};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            globalSize,
            0,
            0, 0, 0
    );
    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}