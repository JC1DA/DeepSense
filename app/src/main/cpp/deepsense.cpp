#include <jni.h>
#include <string>
#include <deepsense_lib.hpp>
#include <deepsense_internal_lib.hpp>
#include <predefine.hpp>
#include <utilities.hpp>
#include <basic_functions.hpp>
#include <classifier.hpp>

cnn *model = NULL;
OpenCLObjects openCLObjects;

OpenCLObjects *getOpenClObject() {
    return &openCLObjects;
}

cnn           *getModel() {
    return model;
}

extern "C" void Java_com_lanytek_deepsensev3_MainActivity_InitGPU(
        JNIEnv* env,
        jobject thiz,
        jstring model_dir_path,
        jstring packageName
) {
    //init GPU first
    const char *packageNameStr = env->GetStringUTFChars(packageName, 0);
    init_OpenCL(CL_DEVICE_TYPE_GPU, openCLObjects, packageNameStr);
    env->ReleaseStringUTFChars(packageName, packageNameStr);

    //init model
    const char *modelPath = env->GetStringUTFChars(model_dir_path, 0);
    if(model != NULL) {
        cnn_free(model);
    }

    model = cnn_loadModel(modelPath, 1);

    env->ReleaseStringUTFChars(model_dir_path, modelPath);
}

extern "C" jfloatArray Java_com_lanytek_deepsensev3_MainActivity_GetInferrence(
        JNIEnv* env,
        jobject thisObject,
        jfloatArray input
) {
    if(model == NULL)
        return NULL;

    cnn_frame *frame = frame_init(model->input_w, model->input_h, model->input_c);
    jfloat* data = env->GetFloatArrayElements(input, 0);
    memcpy(frame->data, data, model->input_w * model->input_h * model->input_c * sizeof(float));
    env->ReleaseFloatArrayElements(input, data, 0);

    float *result = cnn_doClassification(frame, model);

    if(result != NULL) {
        int outputSize = model->layers[model->nLayers - 1].output_w * model->layers[model->nLayers - 1].output_h * model->layers[model->nLayers - 1].output_c;
        jfloatArray resultArr = env->NewFloatArray(outputSize);
        env->SetFloatArrayRegion(resultArr, 0, outputSize, result);
        //may lead to memory leak
        return  resultArr;
    } else
        return NULL;
}

