#include <layers/conv_layer.hpp>
#include <clio.hpp>
#include <basic_functions.hpp>
#include <deepsense_internal_lib.hpp>

cnn_frame *doFeedForward_CONV(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = frame_convert_to_cpu(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    cnn_frame *output = frame_init(\
        (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1, \
        (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1, \
        conv_layer->n);

    int i, j, k, x, y, z;
    for(i = 0 ; i < output->c; i++) {
        for(j = 0 ; j < output->h ; j++) {
            for(k = 0 ; k < output->w ; k++) {
                float result = 0.0f;
                for(x = 0 ; x < conv_layer->c; x++) {
                    for(y = 0 ; y < conv_layer->h; y++) {
                        for(z = 0 ; z < conv_layer->w ; z++) {
                            int w = k * conv_layer->stride[0] - conv_layer->pad[0] + z;
                            int h = j * conv_layer->stride[1] - conv_layer->pad[2] + y;
                            if(w < 0 || w >= frame->w)
                                continue;
                            if(h < 0 || h >= frame->h)
                                continue;

                            float tmp1 = getDataFrom3D(frame->data, frame->h, frame->w, frame->c, h, w, x);
                            float tmp2 = getDataFrom4D(conv_layer->W, conv_layer->n, conv_layer->h, conv_layer->w, conv_layer->c, i, y, z, x);
                            result += tmp1 * tmp2;
                        }
                    }
                }

                result += conv_layer->bias[i];
                output->data[getIndexFrom3D(output->c, output->h, output->w, i, j, k)] = result;
            }
        }
    }

    frame_free(frame);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    return output;
}

cnn_frame *doFeedForward_CONV_GPU(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    OpenCLObjects *openCLObjects = getOpenClObject();
    cl_int err = CL_SUCCESS;

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;

    int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_c = conv_layer->n;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_kernel.kernel : openCLObjects->conv_kernel_float.kernel;

    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem),&conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t localSize[3] = {8 , 8, 1};
    int global_x = ((output->w - 1) / localSize[0] + 1) * localSize[0];
    int global_y = ((output->h - 1) / localSize[1] + 1) * localSize[1];

    int didive = 8;
    int gs3 = output_c % didive == 0 ? output_c / didive : output_c;

    size_t globalSize[3] = {(size_t)global_x, (size_t)global_y, (size_t)gs3};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            3,
            0,
            globalSize,
            localSize,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    frame_free(frame);

    return output;
}

cnn_frame *doFeedForward_CONV_FC_GPU(cnn_frame *frame, void *layer) {
    LOGD("Running function %s", __PRETTY_FUNCTION__);

    frame = ((cnn_layer *)layer)->useHalf ? frame_convert_to_gpu_half(frame) : frame_convert_to_gpu_float(frame);

    cl_int err = CL_SUCCESS;

    cnn_layer_conv *conv_layer = ((cnn_layer *)layer)->conv_layer;
    OpenCLObjects *openCLObjects = getOpenClObject();

    int output_w = (frame->w + conv_layer->pad[0] + conv_layer->pad[1] - conv_layer->w) / conv_layer->stride[0] + 1;
    int output_h = (frame->h + conv_layer->pad[2] + conv_layer->pad[3] - conv_layer->h) / conv_layer->stride[1] + 1;
    int output_c = conv_layer->n;

    cnn_frame *output = ((cnn_layer *)layer)->useHalf ? frame_init_gpu_half(output_w, output_h, output_c) : frame_init_gpu(output_w, output_h, output_c);

    cl_mem cl_frame = frame->cl_data;
    cl_mem cl_result = output->cl_data;

    int i = 0;
    cl_kernel kernel = ((cnn_layer *)layer)->useHalf ? openCLObjects->conv_fc_kernel.kernel : openCLObjects->conv_fc_kernel_float.kernel;
    err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_frame);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &frame->c);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &conv_layer->cl_W);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &conv_layer->cl_bias);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->c);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->n);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->stride[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[0]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[1]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[2]);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &conv_layer->pad[3]);
    err |= clSetKernelArg(kernel, i++, sizeof(cl_mem), &cl_result);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_w);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_h);
    err |= clSetKernelArg(kernel, i++, sizeof(int), &output_c);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    //size_t globalSize[1] = {(size_t)output->c};
    size_t globalSize[1] = {(size_t)(output->c % 256 == 0 ? 256 : output->c)};

    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            globalSize,
            0,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err |= clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    frame_free(frame);

    doFeedForward_Activation(output, ((cnn_layer *)layer)->activation);

    return output;
}

