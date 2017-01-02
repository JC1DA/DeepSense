#include <layers/lrn.hpp>
#include <basic_functions.hpp>
#include <clio.hpp>
#include <math.h>
#include <deepsense_internal_lib.hpp>

cnn_frame *doFeedForward_LRN(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = frame_convert_to_cpu(frame);

    cnn_layer_lrn *lrn_layer = ((cnn_layer *)layer)->lrn_layer;

    int channels = frame->c;
    int width    = frame->w;
    int height   = frame->h;

    cnn_frame *output = frame_init(width, height, channels);

    float alpha_over_size = lrn_layer->alpha / lrn_layer->size;
    int size = lrn_layer->size;
    int k = lrn_layer->k;
    float beta = lrn_layer->beta;

    float *in = frame->data;
    float *out = output->data;

    for(int w = 0 ; w < width ; w++) {
        for(int h = 0 ; h < height ; h++) {
            int offset = (h * width + w) * channels;
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            float accum_scale = 0;

            while (head < post_pad) {
                float data = in[offset + head];
                accum_scale += data * data;
                head++;
            }

            while (head < size) {
                float data = in[offset + head];
                accum_scale += data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels) {
                float data = in[offset + head];
                accum_scale += data * data;
                data = in[offset + head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta);
                head++;
            }

            while (head < channels + post_pad) {
                float data = in[offset + head - size];
                accum_scale -= data * data;
                float scale = k + accum_scale * alpha_over_size;
                out[offset + head - post_pad] = in[offset + head - post_pad] * pow(scale, -beta);
                head++;
            }
        }
    }

    frame_free(frame);
    return output;
}

cnn_frame *doFeedForward_LRN_GPU(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    cnn_layer_lrn *lrn_layer = ((cnn_layer *)layer)->lrn_layer;

    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    int channels = frame->c;
    int width    = frame->w;
    int height   = frame->h;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(width, height, channels) : frame_init_gpu(width, height, channels);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    float alpha_over_size = lrn_layer->alpha / lrn_layer->size;

    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->lrn_kernel.kernel : openCLObjects->lrn_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &channels);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &height);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &width);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &lrn_layer->k);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &lrn_layer->size);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &alpha_over_size);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &lrn_layer->beta);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t globalSize[2] = {(size_t)width, (size_t)height};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            2,
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