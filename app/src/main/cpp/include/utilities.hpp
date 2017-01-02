#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include <CL/cl.h>
#include <predefine.hpp>

void init_OpenCL(
    cl_device_type required_device_type,
    OpenCLObjects& openCLObjects,
    const char *packageName);

void shutdown_OpenCL (OpenCLObjects& openCLObjects);

#endif
