#ifndef CLFFT_HELPER_HPP_
#define CLFFT_HELPER_HPP_

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <clFFT.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <utility> // pair
#include <assert.h>

#define CHECK_CL( err ) gearshifft::ClFFT::check_error( err, #err,  __FILE__, __LINE__ )

#define STRINGIFY(A) #A
#define clFFTStatusCase(s) case s: return STRINGIFY(s)

namespace gearshifft {
  namespace ClFFT {

    inline const char *getOpenCLErrorString(cl_int error) {
      switch(error){
        // run-time and JIT compiler errors
      case 0: return "CL_SUCCESS";
      case -1: return "CL_DEVICE_NOT_FOUND";
      case -2: return "CL_DEVICE_NOT_AVAILABLE";
      case -3: return "CL_COMPILER_NOT_AVAILABLE";
      case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5: return "CL_OUT_OF_RESOURCES";
      case -6: return "CL_OUT_OF_HOST_MEMORY";
      case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8: return "CL_MEM_COPY_OVERLAP";
      case -9: return "CL_IMAGE_FORMAT_MISMATCH";
      case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -11: return "CL_BUILD_PROGRAM_FAILURE";
      case -12: return "CL_MAP_FAILURE";
      case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -15: return "CL_COMPILE_PROGRAM_FAILURE";
      case -16: return "CL_LINKER_NOT_AVAILABLE";
      case -17: return "CL_LINK_PROGRAM_FAILURE";
      case -18: return "CL_DEVICE_PARTITION_FAILED";
      case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
      case -30: return "CL_INVALID_VALUE";
      case -31: return "CL_INVALID_DEVICE_TYPE";
      case -32: return "CL_INVALID_PLATFORM";
      case -33: return "CL_INVALID_DEVICE";
      case -34: return "CL_INVALID_CONTEXT";
      case -35: return "CL_INVALID_QUEUE_PROPERTIES";
      case -36: return "CL_INVALID_COMMAND_QUEUE";
      case -37: return "CL_INVALID_HOST_PTR";
      case -38: return "CL_INVALID_MEM_OBJECT";
      case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40: return "CL_INVALID_IMAGE_SIZE";
      case -41: return "CL_INVALID_SAMPLER";
      case -42: return "CL_INVALID_BINARY";
      case -43: return "CL_INVALID_BUILD_OPTIONS";
      case -44: return "CL_INVALID_PROGRAM";
      case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46: return "CL_INVALID_KERNEL_NAME";
      case -47: return "CL_INVALID_KERNEL_DEFINITION";
      case -48: return "CL_INVALID_KERNEL";
      case -49: return "CL_INVALID_ARG_INDEX";
      case -50: return "CL_INVALID_ARG_VALUE";
      case -51: return "CL_INVALID_ARG_SIZE";
      case -52: return "CL_INVALID_KERNEL_ARGS";
      case -53: return "CL_INVALID_WORK_DIMENSION";
      case -54: return "CL_INVALID_WORK_GROUP_SIZE";
      case -55: return "CL_INVALID_WORK_ITEM_SIZE";
      case -56: return "CL_INVALID_GLOBAL_OFFSET";
      case -57: return "CL_INVALID_EVENT_WAIT_LIST";
      case -58: return "CL_INVALID_EVENT";
      case -59: return "CL_INVALID_OPERATION";
      case -60: return "CL_INVALID_GL_OBJECT";
      case -61: return "CL_INVALID_BUFFER_SIZE";
      case -62: return "CL_INVALID_MIP_LEVEL";
      case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64: return "CL_INVALID_PROPERTY";
      case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
      case -66: return "CL_INVALID_COMPILER_OPTIONS";
      case -67: return "CL_INVALID_LINKER_OPTIONS";
      case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
      case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
      case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
      case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
      case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";

        // CLFFT
        clFFTStatusCase(CLFFT_BUGCHECK);
        clFFTStatusCase(CLFFT_NOTIMPLEMENTED);
        clFFTStatusCase(CLFFT_TRANSPOSED_NOTIMPLEMENTED);
        clFFTStatusCase(CLFFT_FILE_NOT_FOUND);
        clFFTStatusCase(CLFFT_FILE_CREATE_FAILURE);
        clFFTStatusCase(CLFFT_VERSION_MISMATCH);
        clFFTStatusCase(CLFFT_INVALID_PLAN);
        clFFTStatusCase(CLFFT_DEVICE_NO_DOUBLE);
        clFFTStatusCase(CLFFT_DEVICE_MISMATCH);
      default: return "Unknown OpenCL error";
      }
    }

    template<typename T>
    inline void check_error( T err, const char* func, const char *file, const int line ) {
      if ( CL_SUCCESS != err ) {
        throw std::runtime_error("OpenCL error "
                                 + std::string(getOpenCLErrorString(err))
                                 +" ["+std::to_string(err)+"]"
                                 +" "+std::string(file)
                                 +":"+std::to_string(line)
                                 +" "+std::string(func)
          );
      }
    }

    inline cl_ulong getMaxGlobalMemSize(cl_device_id dev_id) {
      cl_ulong value = 0;
      // print device name
      CHECK_CL( clGetDeviceInfo(dev_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &value, NULL) );
      return value;
    }

    inline cl_device_type getDeviceType(cl_device_id dev_id) {
      cl_device_type dev_type;
      CHECK_CL( clGetDeviceInfo(dev_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &dev_type, NULL) );
      return dev_type;
    }

    inline std::stringstream getClDeviceInformations(cl_device_id dev_id) {
      std::stringstream info;
      std::vector<std::pair<std::string,std::string> > values;
      char* value = nullptr;
      size_t valueSize = 0;
      cl_uint maxComputeUnits;
      // print device name
      clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
      value = new char[valueSize];
      clGetDeviceInfo(dev_id, CL_DEVICE_NAME, valueSize, value, NULL);
      values.emplace_back("Device", value);
      delete[] value;

      // print hardware device version
      clGetDeviceInfo(dev_id, CL_DEVICE_VERSION, 0, NULL, &valueSize);
      value = new char[valueSize];
      clGetDeviceInfo(dev_id, CL_DEVICE_VERSION, valueSize, value, NULL);
      values.emplace_back("Hardware", value);
      delete[] value;

      // print software driver version
      clGetDeviceInfo(dev_id, CL_DRIVER_VERSION, 0, NULL, &valueSize);
      value = new char[valueSize];
      clGetDeviceInfo(dev_id, CL_DRIVER_VERSION, valueSize, value, NULL);
      values.emplace_back("Software", value);
      delete[] value;

      // print c version supported by compiler for device
      clGetDeviceInfo(dev_id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
      value = new char[valueSize];
      clGetDeviceInfo(dev_id, CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
      values.emplace_back("OpenCL", value);
      delete[] value;

      // print parallel compute units
      clGetDeviceInfo(dev_id, CL_DEVICE_MAX_COMPUTE_UNITS,
                      sizeof(maxComputeUnits), &maxComputeUnits, NULL);
      values.emplace_back("UsedComputeUnits", std::to_string(maxComputeUnits));

      values.emplace_back("MaxGlobalMemSize", std::to_string( getMaxGlobalMemSize(dev_id) ));

      info << "\"ClFFT Informations\"";
      for(auto pair : values) {
        info << ",\"" << pair.first << "\",\"" << pair.second << '"';
      }

      // clFFT version
      cl_uint  major, minor, patch;
      clfftGetVersion(&major, &minor, &patch);
      info << ",\"clFFT\",\"" << major << "." << minor << "." << patch <<"\"";
      return info;
    }

    inline void findClDevice(cl_device_type devtype,
                             cl_platform_id* platform,
                             cl_device_id* device) {
      cl_uint num_of_platforms = 0, num_of_devices = 0;
      cl_device_id device_id = 0;
      if (clGetPlatformIDs(0, NULL, &num_of_platforms) != CL_SUCCESS)
        throw std::runtime_error("Unable to get platform_id");

      cl_platform_id *platform_ids = new cl_platform_id[num_of_platforms];
      if (clGetPlatformIDs(num_of_platforms, platform_ids, NULL) != CL_SUCCESS) {
        delete[] platform_ids;
        throw std::runtime_error("Unable to get platform id array");
      }
      bool found = false;
      for(cl_uint i=0; i<num_of_platforms; i++) {
        if(clGetDeviceIDs(platform_ids[i],
                          devtype,
                          1,
                          &device_id,
                          &num_of_devices) == CL_SUCCESS){
          found = true;
          *platform = platform_ids[i];
          *device = device_id;
          break;
        }
      }
      delete[] platform_ids;
      if(!found) {
        CHECK_CL(clGetPlatformIDs( 1, platform, NULL ));
        CHECK_CL(clGetDeviceIDs( *platform, CL_DEVICE_TYPE_DEFAULT, 1, device, NULL ));
      }
    }

    inline std::stringstream listClDevices() {
      cl_uint num_of_platforms = 0;
      if (clGetPlatformIDs(0, NULL, &num_of_platforms) != CL_SUCCESS)
        throw std::runtime_error("Unable to get platform_id");

      cl_platform_id *platform_ids = new cl_platform_id[num_of_platforms];
      if (clGetPlatformIDs(num_of_platforms, platform_ids, NULL) != CL_SUCCESS) {
        delete[] platform_ids;
        throw std::runtime_error("Unable to get platform id array");
      }

      std::stringstream info;
      cl_uint num_of_devices = 0;
      for(cl_uint i=0; i<num_of_platforms; i++) {
        CHECK_CL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_of_devices) );
        cl_device_id* devices = new cl_device_id[num_of_devices];
        if(clGetDeviceIDs(platform_ids[i],
                          CL_DEVICE_TYPE_ALL,
                          num_of_devices,
                          devices,
                          NULL) != CL_SUCCESS) {
          delete[] devices;
          throw std::runtime_error("Unable to get devices");
        }

        for(cl_uint j=0; j<num_of_devices; ++j) {
          info << "\"ID\"," << i << ":" << j << "," << getClDeviceInformations(devices[j]).str() << std::endl;
        } // devices
        delete[] devices;
      } // platforms
      delete[] platform_ids;
      return info;
    }

    void getPlatformAndDeviceByID(cl_platform_id* platform_id,
                                  cl_device_id* device_id,
                                  cl_uint id_platform,
                                  cl_uint id_device) {

      cl_uint num_of_platforms = 0;
      if(clGetPlatformIDs(0, NULL, &num_of_platforms) != CL_SUCCESS)
        throw std::runtime_error("Unable to get platform_id");
      if(num_of_platforms <= id_platform)
        throw std::runtime_error("Unable to get platform_id");

      cl_platform_id *platform_ids = new cl_platform_id[id_platform+1];
      CHECK_CL( clGetPlatformIDs(id_platform+1, platform_ids, NULL) );

      cl_uint num_of_devices = 0;
      CHECK_CL( clGetDeviceIDs(platform_ids[id_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_of_devices) );
      if(num_of_devices <= id_device)
        throw std::runtime_error("Unable to get device_id");

      cl_device_id* devices = new cl_device_id[id_device+1];
      CHECK_CL( clGetDeviceIDs(platform_ids[id_platform], CL_DEVICE_TYPE_ALL, id_device+1, devices, NULL) );
      *platform_id = platform_ids[id_platform];
      *device_id   = devices[id_device];
    }
  } // ClFFT
} // gearshifft

#endif
