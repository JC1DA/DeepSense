#include <basic_functions.hpp>
#include <math.h>
#include <malloc.h>
#include <predefine.hpp>
#include <clio.hpp>
#include <string.h>
#include <deepsense_internal_lib.hpp>
#include <sys/time.h>

timestamp_t get_timestamp () {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int getIndexFrom4D(int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
    return i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
}

float getDataFrom4D(float *data, int d1, int d2, int d3, int d4, int i1, int i2, int i3, int i4) {
    int index = i1 * (d2 * d3 * d4) + i2 * (d3 * d4) + i3 * d4 + i4;
    return data[index];
}

int getIndexFrom3D(int d1, int d2, int d3, int i1, int i2, int i3) {
    return i1 * (d2 * d3) + i2 * d3 + i3;
}

float getDataFrom3D(float *data, int d1, int d2, int d3, int i1, int i2, int i3) {
    int index = i1 * (d2 * d3) + i2 * d3 + i3;
    return data[index];
}

cnn_frame *activate_RAMP(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] = x * (x > 0) + 0.1 * x;
    }
    return frame;
}

cnn_frame *activate_LOGISTIC(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] = 1./(1. + exp(-x));
    }
    return frame;
}

cnn_frame *activate_RELU(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] =(x > 0) ? x : 0;
    }
    return frame;
}

cnn_frame *activate_LEAKY(cnn_frame *frame) {
    int i;
    for(i = 0 ; i < frame->c * frame->h * frame->w ; i++) {
        float x = frame->data[i];
        frame->data[i] =(x > 0) ? x : 0.1 * x;
    }
    return frame;
}

cnn_frame *doFeedForward_Activation(cnn_frame *frame, int activation) {
    if(activation == NO_ACTIVATION)
        return frame;

    if(!frame->useGPU) {
        switch(activation) {
            case LOGISTIC:
                activate_LOGISTIC(frame);
                break;
            case RAMP:
                activate_RAMP(frame);
                break;
            case LEAKY:
                activate_LEAKY(frame);
                break;
            case RELU:
                activate_RELU(frame);
                break;
        }
    } else {
        OpenCLObjects *openCLObjects = getOpenClObject();
        cl_int err = CL_SUCCESS;
        int i = 0;

        cl_kernel kernel = (frame->useHalf) ? openCLObjects->activation_kernel.kernel : openCLObjects->activation_kernel_float.kernel;
        err  = clSetKernelArg(kernel, i++, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, i++, sizeof(int), &activation);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t globalSize[1] = {(size_t)(frame->w * frame->h * frame->c)};

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
    }

    return frame;
}

cnn_frame *frame_init(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->data = (float *)calloc(w * h * c, sizeof(float));
    frame->useGPU = 0;
    frame->useHalf = 0;
    return frame;
}

cnn_frame *frame_init_gpu(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 1;
    frame->useHalf = 0;

    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    frame->cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            frame->w * frame->h * frame->c * sizeof(float), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(err == CL_SUCCESS)
        return frame;
    else {
        free(frame);
        return NULL;
    }
}

cnn_frame *frame_init_gpu_half(int w, int h, int c) {
    cnn_frame *frame = (cnn_frame *)calloc(1, sizeof(cnn_frame));
    frame->w = w;
    frame->h = h;
    frame->c = c;
    frame->useGPU = 1;
    frame->useHalf = 1;

    cl_int err;
    OpenCLObjects *openCLObjects = getOpenClObject();

    frame->cl_data = clCreateBuffer(
            openCLObjects->context,
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            frame->w * frame->h * frame->c * sizeof(cl_half), //size in bytes
            NULL,//buffer of data
            &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(err == CL_SUCCESS)
        return frame;
    else {
        free(frame);
        return NULL;
    }
}

cnn_frame       *   frame_clone(cnn_frame *src) {
    if(!src->useGPU) {
        cnn_frame *frame = frame_init(src->w, src->h, src->c);
        memcpy(frame->data, src->data, frame->w * frame->h * frame->c * sizeof(float));
        return frame;
    } else {
        cl_int err = CL_SUCCESS;
        cnn_frame *frame = NULL;
        if(src->useHalf == 0)
            frame = frame_init_gpu(src->w, src->h, src->c);
        else
            frame = frame_init_gpu_half(src->w, src->h, src->c);

        if(frame == NULL)
            return NULL;

        int mapped_size = src->w * src->h * src->c * sizeof(float);
        if(src->useHalf == 1)
            mapped_size = src->w * src->h * src->c * sizeof(cl_half);

        OpenCLObjects *openCLObjects = getOpenClObject();
        float *buf_src = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					src->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					mapped_size, \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        float *buf_dst = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					mapped_size, \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        memcpy((void*)buf_dst, (void*)buf_src, mapped_size);

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					src->cl_data, \
					buf_src, \
					0, NULL, NULL);

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					buf_dst, \
					0, NULL, NULL);
        return frame;
    }
}

cnn_frame* frame_convert_to_gpu_float(cnn_frame *frame) {
    if(frame->useGPU && !frame->useHalf)
        return frame;

    OpenCLObjects *openCLObjects = getOpenClObject();
    cnn_frame *output = frame_init_gpu(frame->w, frame->h, frame->c);
    int err = CL_SUCCESS;

    if(!frame->useGPU) {
        //CPU-mode
        float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					output->cl_data, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					output->w * output->h * output->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        memcpy((void *)buf_dest, frame->data, output->w * output->h * output->c * sizeof(cl_float));

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					output->cl_data, \
					buf_dest, \
					0, NULL, NULL);
    } else {
        //GPU-half-mode
        cl_kernel kernel = openCLObjects->convert_half_to_float_kernel.kernel;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t convertSize[1] = {(size_t) output->w * output->h * output->c};
        err = clEnqueueNDRangeKernel(
                openCLObjects->queue,
                kernel,
                1,
                0,
                convertSize,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    }

    frame_free(frame);

    //test
    {
        float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					output->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					output->w * output->h * output->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					output->cl_data, \
					buf_dest, \
					0, NULL, NULL);
    }

    return output;
}

cnn_frame* frame_convert_to_gpu_half(cnn_frame *frame) {
    if(frame->useGPU && frame->useHalf)
        return frame;

    cnn_frame *output = frame_init_gpu_half(frame->w, frame->h, frame->c);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    cl_mem cl_data = NULL;

    if(!frame->useGPU) {
        //cpu-mode
        cl_data = clCreateBuffer(
                openCLObjects->context,
                CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                frame->w * frame->h * frame->c * sizeof(float), //size in bytes
                frame->data,//buffer of data
                &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
    } else {
        //gpu-float-mode
        cl_data = frame->cl_data;
    }

    cl_kernel kernel = openCLObjects->convert_float_to_half_kernel.kernel;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_data);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output->cl_data);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    size_t convertSize[1] = {(size_t) output->w * output->h * output->c};
    err = clEnqueueNDRangeKernel(
            openCLObjects->queue,
            kernel,
            1,
            0,
            convertSize,
            0,
            0, 0, 0
    );
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    err = clFinish(openCLObjects->queue);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    if(!frame->useGPU)
        clReleaseMemObject(cl_data);

    frame_free(frame);

    //test
    {
        float *buf_dest = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					output->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					output->w * output->h * output->c * sizeof(cl_half), \
					0, NULL, NULL, &err);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        clEnqueueUnmapMemObject(openCLObjects->queue, \
					output->cl_data, \
					buf_dest, \
					0, NULL, NULL);
    }

    return output;
}

cnn_frame * frame_convert_to_cpu(cnn_frame *frame) {
    if(!frame->useGPU)
        return frame;

    cnn_frame *output = frame_init(frame->w, frame->h, frame->c);
    OpenCLObjects *openCLObjects = getOpenClObject();
    int err = CL_SUCCESS;

    //convert half to float first
    if(frame->useHalf) {

        cnn_frame *tmp = frame_init_gpu(frame->w, frame->h, frame->c);

        cl_kernel kernel = openCLObjects->convert_half_to_float_kernel.kernel;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &frame->cl_data);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &tmp->cl_data);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        size_t convertSize[1] = {(size_t) tmp->w * tmp->h * tmp->c};
        err = clEnqueueNDRangeKernel(
                openCLObjects->queue,
                kernel,
                1,
                0,
                convertSize,
                0,
                0, 0, 0
        );
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        err = clFinish(openCLObjects->queue);
        SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

        frame_free(frame);
        frame = tmp;
    }

    //map gpu-mem to cpu-mem and copy data
    float *buf_src = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					frame->cl_data, \
					CL_TRUE, CL_MAP_READ, \
					0, \
					frame->w * frame->h * frame->c * sizeof(cl_float), \
					0, NULL, NULL, &err);
    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

    memcpy((void*)output->data, (void*)buf_src, frame->w * frame->h * frame->c * sizeof(cl_float));

    clEnqueueUnmapMemObject(openCLObjects->queue, \
					frame->cl_data, \
					buf_src, \
					0, NULL, NULL);

    frame_free(frame);

    return output;
}

void frame_free(cnn_frame *frame) {
    if(frame->useGPU == 0)
        free(frame->data);
    else {
        clReleaseMemObject(frame->cl_data);
    }
    free(frame);
}