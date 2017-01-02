#ifndef __CLIO_HPP__
#define __CLIO_HPP__

#include <android/log.h>
#include <CL/cl.h>
#include <predefine.hpp>

// Commonly-defined shortcuts for LogCat output from native C applications.
#define  LOG_TAG    PROGRAM_NAME
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/* This function helps to create informative messages in
 * case when OpenCL errors occur. The function returns a string
 * representation for an OpenCL error code.
 * For example, "CL_DEVICE_NOT_FOUND" instead of "-1".
 */
const char* opencl_error_to_str (cl_int error);

#define SAMPLE_CHECK_ERRORS(ERR)                                                      \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return;                                                                       \
    }

#define SAMPLE_CHECK_ERRORS_WITH_RETURN(ERR) \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return ERR;                                                                   \
    }

#define SAMPLE_CHECK_ERRORS_WITH_NULL_RETURN(ERR) \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGD                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return NULL;                                                                   \
    }    

#endif
