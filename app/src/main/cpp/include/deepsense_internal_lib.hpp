#ifndef __LNN_INTERNAL_LIB_HPP__
#define __LNN_INTERNAL_LIB_HPP__

#ifdef __cplusplus
extern "C" {
#endif

#include <predefine.hpp>

typedef unsigned long long timestamp_t;
timestamp_t get_timestamp();

OpenCLObjects *getOpenClObject();
cnn           *getModel();

#ifdef __cplusplus
}
#endif

#endif
