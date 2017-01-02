#include <layers/maxpool.hpp>
#include <clio.hpp>
#include <basic_functions.hpp>
#include <deepsense_internal_lib.hpp>

cnn_frame *doFeedForward_MAXPOOL(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = frame_convert_to_cpu(frame);

    cnn_layer_maxpool *maxpool_layer = ((cnn_layer *)layer)->maxpool_layer;

    int w = 1 + (frame->w + maxpool_layer->pad[0] + maxpool_layer->pad[1] - maxpool_layer->size) / maxpool_layer->stride[0];
    int h = 1 + (frame->h + maxpool_layer->pad[2] + maxpool_layer->pad[3] - maxpool_layer->size) / maxpool_layer->stride[1];
    int d = frame->c;
    cnn_frame *output = frame_init(w, h, d);

    for(int k = 0 ; k < output->c ; k++) {
        for(int h = 0; h < output->h; h++) {
            for(int w = 0; w < output->w ; w++) {
                float max = -999999.9f;
                for(int x = 0 ; x < maxpool_layer->size ; x++) {
                    for(int y = 0 ; y < maxpool_layer->size ; y++) {
                        int x_ = w * maxpool_layer->stride[0] + x - maxpool_layer->pad[0];
                        int y_ = h * maxpool_layer->stride[1] + y - maxpool_layer->pad[2];
                        int valid = (x_ >= 0 && x_ < frame->w && y_ >= 0 && y_ < frame->h);
                        float val = (valid != 0) ? frame->data[getIndexFrom3D(frame->h, frame->w, frame->c, y_, x_, k)] : -999999.9f;
                        max   = (val > max) ? val   : max;
                    }
                }
                output->data[getIndexFrom3D(output->h, output->w, output->c, h, w, k)] = max;
            }
        }
    }

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_MAXPOOL_GPU(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_maxpool *maxpool_layer = ((cnn_layer *)layer)->maxpool_layer;

    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    int input_w = frame->w;
    int input_h = frame->h;
    int input_d = frame->c;

    //prepare output
    int output_w = 1 + (frame->w + maxpool_layer->pad[0] + maxpool_layer->pad[1] - maxpool_layer->size) / maxpool_layer->stride[0];
    int output_h = 1 + (frame->h + maxpool_layer->pad[2] + maxpool_layer->pad[3] - maxpool_layer->size) / maxpool_layer->stride[1];
    int output_c = frame->c;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    int i = 0;

    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->maxpool_kernel.kernel : openCLObjects->maxpool_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &input_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &input_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &input_d);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->size);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &maxpool_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t globalSize[3] = {(size_t)output_w, (size_t)output_h, (size_t)output_c};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            0,
            0, 0, 0
    );
    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    frame_free(frame);

    return output;
}
