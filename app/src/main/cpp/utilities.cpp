#include <CL/cl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>

#include <utilities.hpp>
#include <clio.hpp>
#include <predefine.hpp>

char packageNameBuf[256];

inline char* load_Program(const char *inputPath) {
    FILE *fp = fopen(inputPath,"r");
    int fd = fileno(fp);
    struct stat buf;
    fstat(fd, &buf);
    int size = buf.st_size;

    char *buffer = (char *)malloc(size + 1);
    buffer[size] = '\0';
    fread(buffer, size, 1, fp);
    fclose(fp);

    return buffer;
}

cl_int compile_Program(OpenCLObjects& openCLObjects, const char* kernelFileName) {

    std::string kernelPath;
    cl_int err = CL_SUCCESS;

    kernelPath.append("/data/data/");
    kernelPath.append(packageNameBuf);
    kernelPath.append("/app_execdir/");
    kernelPath.append(kernelFileName);

    char* tmp = load_Program(kernelPath.c_str());
    std::string tmpStr(tmp);
    const char *kernelSource = tmpStr.c_str();
    free(tmp);

    openCLObjects.program =
	    clCreateProgramWithSource (
	        openCLObjects.context,
	        1,
	        &kernelSource,
	        0,
	        &err
	    );
	SAMPLE_CHECK_ERRORS_WITH_RETURN(err);

    err = clBuildProgram(openCLObjects.program, 0, 0, "-O3 -cl-mad-enable -cl-fast-relaxed-math", 0, 0);

    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_length = 0;

        err = clGetProgramBuildInfo(
        openCLObjects.program,
        openCLObjects.device,
        CL_PROGRAM_BUILD_LOG,
        0,
        0,
        &log_length);
        SAMPLE_CHECK_ERRORS_WITH_RETURN(err);

        //vector<char> log(log_length);
        char* logbuf = (char*)malloc(log_length);

        err = clGetProgramBuildInfo(
            openCLObjects.program,
            openCLObjects.device,
            CL_PROGRAM_BUILD_LOG,
            log_length,
            (void*)logbuf,
            0);
        SAMPLE_CHECK_ERRORS_WITH_RETURN(err);

        LOGE("Error happened during the build of OpenCL program.\nBuild log:%s", logbuf);
        
        free(logbuf);
    }

        //SAMPLE_CHECK_ERRORS_WITH_RETURN(err);
    return CL_SUCCESS;
}

void init_OpenCL(
    cl_device_type required_device_type,
    OpenCLObjects& openCLObjects,
    const char *packageName) {

	using namespace std;
	cl_int err = CL_SUCCESS;

    LOGD("init_OpenCL: Initializing GPU\n");

	strcpy(packageNameBuf, packageName);

	cl_uint num_of_platforms = 0;
    err = clGetPlatformIDs(0, 0, &num_of_platforms);
    SAMPLE_CHECK_ERRORS(err);

    vector<cl_platform_id> platforms(num_of_platforms);
    // Get IDs for all platforms.
    err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
    SAMPLE_CHECK_ERRORS(err);

    cl_uint i = 0;
    size_t platform_name_length = 0;
    err = clGetPlatformInfo(
                        platforms[i],
                        CL_PLATFORM_NAME,
                        0,
                        0,
                        &platform_name_length
                    );
    SAMPLE_CHECK_ERRORS(err);

    vector<char> platform_name(platform_name_length);
    err = clGetPlatformInfo(
                        platforms[i],
                        CL_PLATFORM_NAME,
                        platform_name_length,
                        &platform_name[0],
                        0
                    );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.platform = platforms[0];

    cl_context_properties context_props[] = {
                    CL_CONTEXT_PLATFORM,
                    cl_context_properties(openCLObjects.platform),
                    0};

    openCLObjects.context = clCreateContextFromType(
            context_props,
            required_device_type,
            0,
            0,
             &err
        );
    SAMPLE_CHECK_ERRORS(err);

    err = clGetContextInfo(
        openCLObjects.context,
        CL_CONTEXT_DEVICES,
        sizeof(openCLObjects.device),
        &openCLObjects.device,
        0);
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.queue = clCreateCommandQueue (
        openCLObjects.context,
        openCLObjects.device,
        0,    // Creating queue properties, refer to the OpenCL specification for details.
        &err);
    SAMPLE_CHECK_ERRORS(err);

    err = compile_Program(openCLObjects, PROGRAM_KERNEL_NAME);
    SAMPLE_CHECK_ERRORS(err);

    cl_device_local_mem_type local_mem_type;
    clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_LOCAL_MEM_TYPE,
            sizeof(cl_device_local_mem_type),
            &local_mem_type,
            NULL
    );
    LOGD("CL_DEVICE_LOCAL_MEM_TYPE %u", local_mem_type);

    openCLObjects.conv_kernel.kernel = clCreateKernel(openCLObjects.program, "conv_kernel_half", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.conv_kernel.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.conv_kernel.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.conv_kernel_float.kernel = clCreateKernel(openCLObjects.program, "conv_kernel_float", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.conv_kernel_float.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.conv_kernel_float.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.maxpool_kernel.kernel = clCreateKernel(openCLObjects.program, "maxpool_kernel_half", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.maxpool_kernel.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.maxpool_kernel.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.maxpool_kernel_float.kernel = clCreateKernel(openCLObjects.program, "maxpool_kernel_float", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.maxpool_kernel_float.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.maxpool_kernel_float.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.conv_fc_kernel.kernel = clCreateKernel(openCLObjects.program, "conv_fc_kernel_half", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.conv_fc_kernel.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.conv_fc_kernel.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.conv_fc_kernel_float.kernel = clCreateKernel(openCLObjects.program, "conv_fc_kernel_float", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.conv_fc_kernel_float.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.conv_fc_kernel_float.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.fully_connected_kernel.kernel = clCreateKernel(openCLObjects.program, "fully_connected_kernel_half", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.fully_connected_kernel.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.fully_connected_kernel.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.fully_connected_kernel_float.kernel = clCreateKernel(openCLObjects.program, "fully_connected_kernel_float", &err);
    SAMPLE_CHECK_ERRORS(err);
    clGetKernelWorkGroupInfo(
            openCLObjects.fully_connected_kernel_float.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.fully_connected_kernel_float.kernel_max_workgroup_size,
            NULL
    );

    openCLObjects.lrn_kernel.kernel = clCreateKernel(openCLObjects.program, "cross_channels_lrn_kernel_half", &err);
    clGetKernelWorkGroupInfo(
            openCLObjects.lrn_kernel.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.lrn_kernel.kernel_max_workgroup_size,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.lrn_kernel_float.kernel = clCreateKernel(openCLObjects.program, "cross_channels_lrn_kernel_float", &err);
    clGetKernelWorkGroupInfo(
            openCLObjects.lrn_kernel_float.kernel,
            openCLObjects.device,
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &openCLObjects.lrn_kernel_float.kernel_max_workgroup_size,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.activation_kernel.kernel = clCreateKernel(openCLObjects.program, "activation_kernel_half", &err);
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.activation_kernel_float.kernel = clCreateKernel(openCLObjects.program, "activation_kernel_float", &err);
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.convert_float_to_half_kernel.kernel = clCreateKernel(openCLObjects.program, "convertFloatToHalf", &err);
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.convert_half_to_float_kernel.kernel = clCreateKernel(openCLObjects.program, "convertHalfToFloat", &err);
    SAMPLE_CHECK_ERRORS(err);

    LOGD("initOpenCL finished successfully");
}

void shutdown_OpenCL (OpenCLObjects& openCLObjects) {
    cl_int err = CL_SUCCESS;

    err = clReleaseKernel(openCLObjects.conv_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.conv_fc_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.maxpool_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.lrn_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.fully_connected_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.conv_kernel_float.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.conv_fc_kernel_float.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.maxpool_kernel_float.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.lrn_kernel_float.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.fully_connected_kernel_float.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.convert_float_to_half_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseKernel(openCLObjects.convert_half_to_float_kernel.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseProgram(openCLObjects.program);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseCommandQueue(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseContext(openCLObjects.context);
    SAMPLE_CHECK_ERRORS(err);

    LOGD("shutdownOpenCL finished successfully");
}