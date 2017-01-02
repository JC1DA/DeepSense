//
// Created by JC1DA on 6/3/16.
//

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <deepsense_lib.hpp>
#include <predefine.hpp>
#include <deepsense_internal_lib.hpp>
#include <clio.hpp>
#include <basic_functions.hpp>
#include <layers/conv_layer.hpp>
#include <layers/fully_connected.hpp>
#include <layers/maxpool.hpp>
#include <layers/softmax.hpp>
#include <layers/lrn.hpp>

static inline int CMP_OPTION(char *str, const char *option) {
    int ret = strncmp(str, option, strlen(option)) == 0 ? 1 : 0;
    return ret;
}

static inline int PARSE_ACTIVATION(char *line) {
    char buf[32];
    sscanf(line,"ACTIVATION: %s\n",buf);
    if(CMP_OPTION(buf, "RAMP"))
        return RAMP;
    else if(CMP_OPTION(buf, "LOGISTIC"))
        return LOGISTIC;
    else if(CMP_OPTION(buf, "LEAKY"))
        return LEAKY;
    else if(CMP_OPTION(buf, "RELU"))
        return RELU;
    return NO_ACTIVATION;
}

cnn *cnn_loadModel(const char *modelDirPath, int useGPU) {
    cnn *model = (cnn *)calloc(1, sizeof(cnn));
    model->useGPU = useGPU;

    {
        /* read number of layers */
        char fileNameBuf[256];
        char line[256];
        sprintf(fileNameBuf,"%s/model",modelDirPath);

        FILE *fp = fopen(fileNameBuf,"r");
        //PLEASE FILL IN NEW FORMAT
        while(fgets(line, sizeof(line), fp)) {
            if(CMP_OPTION(line, "NUMLAYERS"))
                sscanf(line, "NUMLAYERS: %d\n", &model->nLayers);
            else if(CMP_OPTION(line, "W"))
                sscanf(line, "W: %d\n", &model->input_w);
            else if(CMP_OPTION(line, "H"))
                sscanf(line, "H: %d\n", &model->input_h);
            else if(CMP_OPTION(line, "C"))
                sscanf(line, "C: %d\n", &model->input_c);
        }
        fclose(fp);
    }

    model->layers = (cnn_layer *)calloc(model->nLayers, sizeof(cnn_layer));
    cnn_layer *layers = model->layers;

    for(int i = 1 ; i <= model->nLayers ; i++) {
        char fileNameBuf[256];
        char line[256];

        sprintf(fileNameBuf,"%s/l_%d",modelDirPath,i);

        cnn_layer *layer = &layers[i - 1];
        layer->index = i - 1;
        layer->useGPU = model->useGPU;
        layer->type = LAYER_TYPE_UNKNOWN;
        layer->activation = NO_ACTIVATION;

        LOGD("Loading layer %d\n", i);

        FILE *layerfp = fopen(fileNameBuf,"r");
        while (fgets(line, sizeof(line), layerfp)) {
            if(layer->type == LAYER_TYPE_UNKNOWN) {
                if(CMP_OPTION(line, "CONV")) {
                    layer->type = LAYER_TYPE_CONV;
                    layer->conv_layer = (cnn_layer_conv *) calloc(1, sizeof(cnn_layer_conv));
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_CONV;
                    else {
                        layer->doFeedForward = doFeedForward_CONV_GPU;
                    }
                    layer->conv_layer->group = 1;
                } else if(CMP_OPTION(line, "FULLY_CONNECTED")) {
                    layer->type = LAYER_TYPE_FULLY_CONNECTED;
                    layer->connected_layer = (cnn_layer_fully_connected *) calloc(1,
                                                                                  sizeof(cnn_layer_fully_connected));
                    layer->connected_layer->need_reshape = 0;
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_FULLY_CONNECTED;
                    else
                        layer->doFeedForward = doFeedForward_FULLY_CONNECTED_GPU;
                } else if(CMP_OPTION(line, "MAXPOOL")) {
                    layer->type = LAYER_TYPE_MAXPOOL;
                    layer->maxpool_layer = (cnn_layer_maxpool *) calloc(1,
                                                                        sizeof(cnn_layer_maxpool));
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_MAXPOOL;
                    else {
                        layer->doFeedForward = doFeedForward_MAXPOOL_GPU;
                    }
                } else if(CMP_OPTION(line, "SOFTMAX")) {
                    layer->type = LAYER_TYPE_SOFTMAX;
                    layer->doFeedForward = doFeedForward_SOFTMAX;
                } else if(CMP_OPTION(line, "LRN_NORM")) {
                    layer->type = LAYER_TYPE_LRN_NORMALIZE;
                    layer->lrn_layer = (cnn_layer_lrn *) malloc(sizeof(cnn_layer_lrn));
                    layer->lrn_layer->k = 1;
                    if (!layer->useGPU)
                        layer->doFeedForward = doFeedForward_LRN;
                    else
                        layer->doFeedForward = doFeedForward_LRN_GPU;
                }
            } else {

                if(CMP_OPTION(line, "USE_HALF")) {
                    sscanf(line,"USE_HALF: %d", &layer->useHalf);
                    if(layer->useHalf != 0)
                        layer->useHalf = 1;
                }


                switch(layer->type) {
                    case LAYER_TYPE_CONV:
                        if(CMP_OPTION(line, "STRIDE")) {
                            sscanf(line, "STRIDE: %d %d\n", \
                                &layer->conv_layer->stride[0], \
                                &layer->conv_layer->stride[1]);
                        } else if(CMP_OPTION(line, "PAD")) {
                            sscanf(line,"PAD: %d %d %d %d\n", \
								&layer->conv_layer->pad[0], \
								&layer->conv_layer->pad[1], \
								&layer->conv_layer->pad[2], \
								&layer->conv_layer->pad[3]);
                        } else if(CMP_OPTION(line, "WIDTH")) {
                            sscanf(line,"WIDTH: %d\n",&layer->conv_layer->w);
                        } else if(CMP_OPTION(line, "HEIGHT")) {
                            sscanf(line,"HEIGHT: %d\n",&layer->conv_layer->h);
                        } else if(CMP_OPTION(line, "IN_CHANNELS")) {
                            sscanf(line,"IN_CHANNELS: %d\n",&layer->conv_layer->c);
                        } else if(CMP_OPTION(line, "OUT_CHANNELS")) {
                            sscanf(line,"OUT_CHANNELS: %d\n",&layer->conv_layer->n);
                        } else if(CMP_OPTION(line, "ACTIVATION")) {
                            layer->activation = PARSE_ACTIVATION(line);
                        } else if(CMP_OPTION(line, "GROUP")) {
                            sscanf(line,"GROUP: %d\n",&layer->conv_layer->group);
                        }
                        break;
                    case LAYER_TYPE_FULLY_CONNECTED:
                        if(CMP_OPTION(line, "INPUTSIZE")) {
                            sscanf(line, "INPUTSIZE: %d\n", &layer->connected_layer->inputSize);
                        } else if(CMP_OPTION(line, "OUTPUTSIZE")) {
                            sscanf(line,"OUTPUTSIZE: %d\n", &layer->connected_layer->outputSize);
                        } else if(CMP_OPTION(line, "ACTIVATION")) {
                            layer->activation = PARSE_ACTIVATION(line);
                        } else if(CMP_OPTION(line, "RESHAPE")) {
                            sscanf(line,"RESHAPE: %d\n",&layer->connected_layer->need_reshape);
                        }
                        break;
                    case LAYER_TYPE_MAXPOOL:
                        if(CMP_OPTION(line, "SIZE")) {
                            sscanf(line,"SIZE: %d\n", &layer->maxpool_layer->size);
                        } else if(CMP_OPTION(line, "STRIDE")) {
                            sscanf(line,"STRIDE: %d %d\n", &layer->maxpool_layer->stride[0], &layer->maxpool_layer->stride[1]);
                        } else if(CMP_OPTION(line, "PAD")) {
                            sscanf(line,"PAD: %d %d %d %d\n", &layer->maxpool_layer->pad[0], &layer->maxpool_layer->pad[1], \
                                                            &layer->maxpool_layer->pad[2], &layer->maxpool_layer->pad[3]);
                        }
                        break;
                    case LAYER_TYPE_LRN_NORMALIZE:
                        if(CMP_OPTION(line, "SIZE")) {
                            sscanf(line,"SIZE: %d\n", &layer->lrn_layer->size);
                        } else if(CMP_OPTION(line, "ALPHA")) {
                            sscanf(line,"ALPHA: %f\n", &layer->lrn_layer->alpha);
                        } else if(CMP_OPTION(line, "BETA")) {
                            sscanf(line,"BETA: %f\n", &layer->lrn_layer->beta);
                        }
                        break;
                    case LAYER_TYPE_SOFTMAX:
                        break;
                    case LAYER_TYPE_UNKNOWN:
                        break;
                }
            }
        }
        fclose(layerfp);

        if(layer->type == LAYER_TYPE_CONV) {
            //determine output size
            if(layer->index == 0) {
                layer->output_w = (model->input_w + \
                                            layer->conv_layer->pad[0] + \
                                            layer->conv_layer->pad[1] - \
                                            layer->conv_layer->w) / \
					                        layer->conv_layer->stride[0] + 1;
                layer->output_h = (model->input_h + \
                                            layer->conv_layer->pad[2] + \
                                            layer->conv_layer->pad[3] - \
                                            layer->conv_layer->h) / \
                                            layer->conv_layer->stride[1] + 1;
                layer->output_c = layer->conv_layer->n;
            } else {
                layer->output_w = (layers[layer->index - 1].output_w + \
                                            layer->conv_layer->pad[0] + \
                                            layer->conv_layer->pad[1] - \
                                            layer->conv_layer->w) / \
                                            layer->conv_layer->stride[0] + 1;
                layer->output_h = (layers[layer->index - 1].output_h + \
                                            layer->conv_layer->pad[2] + \
                                            layer->conv_layer->pad[3] - \
                                            layer->conv_layer->h) / \
                                            layer->conv_layer->stride[1] + 1;
                layer->output_c = layer->conv_layer->n;
            }

            //switch to another kernel if this conv layer is equivalent to fully connected layer
            if(layer->output_h == 1 && layer->output_w == 1 && model->useGPU) {
                layer->doFeedForward = doFeedForward_CONV_FC_GPU;
            }

            //LOAD BIAS & WEIGHTS DATA
            char biasFilePath[256];
            strcpy(biasFilePath, fileNameBuf);
            strcat(biasFilePath, "_bias");
            FILE *biasfp = fopen(biasFilePath, "r");
            if(!layer->useGPU) {
                layer->conv_layer->bias = (float *)calloc(layer->conv_layer->n, sizeof(float));
                fread(layer->conv_layer->bias, sizeof(float), layer->conv_layer->n, biasfp);
            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->conv_layer->cl_bias = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->conv_layer->n * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->conv_layer->cl_bias, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->conv_layer->n * sizeof(float), \
					0, NULL, NULL, NULL);

                fread(mappedBuffer, sizeof(float), layer->conv_layer->n, biasfp);

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->conv_layer->cl_bias, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf) {
                    cl_mem cl_bias_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->n * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->conv_layer->cl_bias);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_bias_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t) layer->conv_layer->n};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);
                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->conv_layer->cl_bias);

                    layer->conv_layer->cl_bias = cl_bias_half;
                }
            }
            fclose(biasfp);

            char wFilePath[256];
            strcpy(wFilePath, fileNameBuf);
            strcat(wFilePath, "_weight");
            FILE *wfp = fopen(wFilePath, "r");
            if(!layer->useGPU) {
                layer->conv_layer->W = (float *) calloc(\
                    layer->conv_layer->w * \
                    layer->conv_layer->h * \
                    layer->conv_layer->c * \
                    layer->conv_layer->n / layer->conv_layer->group, sizeof(float));

                /*
                 * Our old model format is [n x c x h x w]
                 * We change memory layout from [n x c x h x w] into [n x h x w x c]
                 */

                float *buffer = (float *)malloc(layer->conv_layer->h * layer->conv_layer->w * sizeof(float));
                for(int k = 0 ; k < layer->conv_layer->n / layer->conv_layer->group ; k++) {
                    for(int c = 0 ; c < layer->conv_layer->c ; c++) {
                        fread(buffer, sizeof(float), layer->conv_layer->h * layer->conv_layer->w, wfp);
                        for(int h = 0 ; h < layer->conv_layer->h ; h++) {
                            for(int w = 0 ; w < layer->conv_layer->w ; w++) {
                                int buf_idx = h * layer->conv_layer->w + w;
                                int new_idx = getIndexFrom4D(
                                        layer->conv_layer->n,
                                        layer->conv_layer->h,
                                        layer->conv_layer->w,
                                        layer->conv_layer->c,
                                        k, h, w, c
                                );
                                layer->conv_layer->W[new_idx] = buffer[buf_idx];
                            }
                        }
                    }
                }
                free(buffer);
            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->conv_layer->cl_W = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n  / layer->conv_layer->group * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->conv_layer->cl_W, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n  / layer->conv_layer->group * sizeof(float), \
					0, NULL, NULL, NULL);

                float *buffer = (float *)malloc(layer->conv_layer->h * layer->conv_layer->w * sizeof(float));
                for(int k = 0 ; k < layer->conv_layer->n  / layer->conv_layer->group ; k++) {
                    for(int c = 0 ; c < layer->conv_layer->c ; c++) {
                        fread(buffer, sizeof(float), layer->conv_layer->h * layer->conv_layer->w, wfp);
                        for(int h = 0 ; h < layer->conv_layer->h ; h++) {
                            for(int w = 0 ; w < layer->conv_layer->w ; w++) {
                                int buf_idx = h * layer->conv_layer->w + w;
                                int new_idx = getIndexFrom4D(
                                        layer->conv_layer->n,
                                        layer->conv_layer->h,
                                        layer->conv_layer->w,
                                        layer->conv_layer->c,
                                        k, h, w, c
                                );
                                mappedBuffer[new_idx] = buffer[buf_idx];
                            }
                        }
                    }
                }

                free(buffer);

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->conv_layer->cl_W, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf == 1) {
                    cl_mem cl_W_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n  / layer->conv_layer->group * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->conv_layer->cl_W);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_W_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t)layer->conv_layer->w * layer->conv_layer->h * layer->conv_layer->c * layer->conv_layer->n  / layer->conv_layer->group};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->conv_layer->cl_W);

                    layer->conv_layer->cl_W = cl_W_half;
                }
            }
            fclose(wfp);
        }

        if(layer->type == LAYER_TYPE_FULLY_CONNECTED) {
            layer->output_w = 1;
            layer->output_h = 1;
            layer->output_c = layer->connected_layer->outputSize;

            layer->connected_layer->weightSize = layer->connected_layer->inputSize * layer->connected_layer->outputSize;

            //LOAD BIAS AND WEIGHTS DATA
            char biasFilePath[256];
            strcpy(biasFilePath, fileNameBuf);
            strcat(biasFilePath, "_bias");
            FILE *biasfp = fopen(biasFilePath, "r");
            if(!layer->useGPU) {
                layer->connected_layer->bias = (float *)calloc(layer->connected_layer->outputSize, sizeof(float));
                fread(layer->connected_layer->bias, sizeof(float), layer->connected_layer->outputSize, biasfp);
            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->connected_layer->cl_bias = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->connected_layer->outputSize * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->connected_layer->cl_bias, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->connected_layer->outputSize * sizeof(float), \
					0, NULL, NULL, NULL);

                fread(mappedBuffer, sizeof(float), layer->connected_layer->outputSize, biasfp);

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->connected_layer->cl_bias, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf) {
                    cl_mem cl_bias_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->connected_layer->outputSize * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->connected_layer->cl_bias);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_bias_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t) layer->connected_layer->outputSize};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->connected_layer->cl_bias);

                    layer->connected_layer->cl_bias = cl_bias_half;
                }
            }
            fclose(biasfp);

            char wFilePath[256];
            strcpy(wFilePath, fileNameBuf);
            strcat(wFilePath, "_weight");
            FILE *wfp = fopen(wFilePath, "r");
            if(!layer->useGPU) {
                layer->connected_layer->W = (float *) calloc(\
                    layer->connected_layer->weightSize, sizeof(float));
                fread(layer->connected_layer->W, sizeof(float),
                      layer->connected_layer->weightSize,
                      wfp);
            } else {
                cl_int err;
                OpenCLObjects *openCLObjects = getOpenClObject();

                layer->connected_layer->cl_W = clCreateBuffer(
                        openCLObjects->context,
                        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                        layer->connected_layer->weightSize * sizeof(float), //size in bytes
                        NULL,//buffer of data
                        &err);
                SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                float *mappedBuffer = (float *)clEnqueueMapBuffer(openCLObjects->queue, \
					layer->connected_layer->cl_W, \
					CL_TRUE, CL_MAP_WRITE, \
					0, \
					layer->connected_layer->weightSize * sizeof(float), \
					0, NULL, NULL, NULL);

                if(!layer->connected_layer->need_reshape) {
                    //file is formatted [(c x h x w) x outputsize]
                    //this is for LRCN
                    float * buffer = (float *)malloc(layer->connected_layer->outputSize * sizeof(float));
                    int input_h = (layer->index == 0) ? model->input_h : layers[layer->index - 1].output_h;
                    int input_w = (layer->index == 0) ? model->input_w : layers[layer->index - 1].output_w;
                    int input_c = (layer->index == 0) ? model->input_c : layers[layer->index - 1].output_c;

                    for(int c = 0 ; c < input_c ; c++) {
                        for(int h = 0 ; h < input_h ; h++) {
                            for(int w = 0 ; w < input_w ; w++) {
                                fread(buffer, sizeof(float), layer->connected_layer->outputSize, wfp);
                                for(int n = 0 ; n < layer->connected_layer->outputSize ; n++) {
                                    int idx = getIndexFrom4D(layer->connected_layer->outputSize, input_h, input_w, input_c, n, h, w, c);
                                    mappedBuffer[idx] = buffer[n];
                                }
                            }
                        }
                    }
                    free(buffer);
                } else {
                    //file is formatted [outputsize x (c x h x w)]
                    int input_h = (layer->index == 0) ? model->input_h : layers[layer->index - 1].output_h;
                    int input_w = (layer->index == 0) ? model->input_w : layers[layer->index - 1].output_w;
                    int input_c = (layer->index == 0) ? model->input_c : layers[layer->index - 1].output_c;

                    int size = input_h * input_w * input_c;
                    float *buffer = (float *)malloc(size * sizeof(float));
                    for(int n = 0 ; n < layer->connected_layer->outputSize ; n++) {
                        fread(buffer, sizeof(float), size, wfp); //[c x h x w]
                        //need to convert to h x w x c
                        int f_idx = 0;
                        for(int c = 0 ; c < input_c ; c++) {
                            for(int h = 0 ; h < input_h ; h++) {
                                for(int w = 0 ; w < input_w ; w++) {
                                    int idx = getIndexFrom4D(layer->connected_layer->outputSize, input_h, input_w, input_c, n, h, w, c);
                                    mappedBuffer[idx] = buffer[f_idx];
                                    f_idx++;
                                }
                            }
                        }
                    }
                    free(buffer);
                }

                clEnqueueUnmapMemObject(openCLObjects->queue, \
					layer->connected_layer->cl_W, \
					mappedBuffer, \
					0, NULL, NULL);

                if(layer->useHalf) {
                    cl_mem cl_W_half = clCreateBuffer(
                            openCLObjects->context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            layer->connected_layer->weightSize * sizeof(cl_half), //size in bytes
                            NULL,//buffer of data
                            &err);

                    err  = clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 0, sizeof(cl_mem), &layer->connected_layer->cl_W);
                    err |= clSetKernelArg(openCLObjects->convert_float_to_half_kernel.kernel, 1, sizeof(cl_mem), &cl_W_half);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    size_t convertSize[1] = {(size_t) layer->connected_layer->weightSize};
                    err = clEnqueueNDRangeKernel(
                            openCLObjects->queue,
                            openCLObjects->convert_float_to_half_kernel.kernel,
                            1,
                            0,
                            convertSize,
                            0,
                            0, 0, 0
                    );
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    err = clFinish(openCLObjects->queue);
                    SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(err);

                    clReleaseMemObject(layer->connected_layer->cl_W);

                    layer->connected_layer->cl_W = cl_W_half;
                }
            }
            fclose(wfp);
        }

        if(layer->type == LAYER_TYPE_MAXPOOL) {
            layer->output_w = 1 + (layers[layer->index - 1].output_w + layer->maxpool_layer->pad[0] + layer->maxpool_layer->pad[1] - layer->maxpool_layer->size) / layer->maxpool_layer->stride[0];
            layer->output_h = 1 + (layers[layer->index - 1].output_h + layer->maxpool_layer->pad[2] + layer->maxpool_layer->pad[3] - layer->maxpool_layer->size) / layer->maxpool_layer->stride[1];
            layer->output_c = layers[layer->index - 1].output_c;
        }

        if(layer->type == LAYER_TYPE_SOFTMAX) {
            layer->output_w = 1;
            layer->output_h = 1;
            layer->output_c = layers[layer->index - 1].output_c;
        }

        if(layer->type == LAYER_TYPE_LRN_NORMALIZE) {
            layer->output_w = layers[layer->index - 1].output_w;
            layer->output_h = layers[layer->index - 1].output_h;
            layer->output_c = layers[layer->index - 1].output_c;
        }

        int input_w = (i == 1) ? model->input_w : layers[i - 2].output_w;
        int input_h = (i == 1) ? model->input_h : layers[i - 2].output_h;
        int input_c = (i == 1) ? model->input_c : layers[i - 2].output_c;

        LOGD("Layer %d has input[%d %d %d] and output [%d %d %d]",(i), \
             input_c, input_h, input_w,
              layer->output_c, layer->output_h, layer->output_w);
    }

    return model;
}

void cnn_free(cnn *model) {
    int i;
    for(i = 0 ; i < model->nLayers ; i++) {
        cnn_layer *layer = &model->layers[i];
        if(layer->type == LAYER_TYPE_CONV) {
            if(!model->useGPU) {
                free(layer->conv_layer->bias);
                free(layer->conv_layer->W);
            } else {
                clReleaseMemObject(layer->conv_layer->cl_W);
                clReleaseMemObject(layer->conv_layer->cl_bias);
            }
            free(layer->conv_layer);
        } else if(layer->type == LAYER_TYPE_FULLY_CONNECTED) {
            if(!model->useGPU) {
                free(layer->connected_layer->bias);
                free(layer->connected_layer->W);
            } else {
                clReleaseMemObject(layer->conv_layer->cl_W);
                clReleaseMemObject(layer->conv_layer->cl_bias);
            }
            free(layer->connected_layer);
        } else if(layer->type == LAYER_TYPE_MAXPOOL) {
            free(layer->maxpool_layer);
        } else if(layer->type == LAYER_TYPE_LRN_NORMALIZE) {
            free(layer->lrn_layer);
        }
    }

    if(model->averageImage != NULL)
        free(model->averageImage);

    free(model->layers);
    free(model);
}