void OpenCL_init()
{
    read_cl_file();
    
    cl_initialization();
        
    cl_load_prog();
}

const char *getOpenCLErrorString(cl_int err) {

   switch(err) {

      case CL_SUCCESS: return "CL_SUCCESS";
      case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
      case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
      case CL_COMPILER_NOT_AVAILABLE: return
                                       "CL_COMPILER_NOT_AVAILABLE";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE: return
                                       "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
      case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
      case CL_PROFILING_INFO_NOT_AVAILABLE: return
                                       "CL_PROFILING_INFO_NOT_AVAILABLE";
      case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
      case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED: return
                                       "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
      case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
      case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
      case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
      case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
      case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
      case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
      case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
      case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
      case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
      case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return
                                       "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
      case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
      case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
      case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
      case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
      case CL_INVALID_PROGRAM_EXECUTABLE: return
                                       "CL_INVALID_PROGRAM_EXECUTABLE";
      case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
      case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
      case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
      case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
      case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
      case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
      case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
      case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
      case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
      case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
      case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
      case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
      case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
      case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
      case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
      case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
      case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
      case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";

      case CLFFT_BUGCHECK: return "CLFFT_BUGCHECK";
      case CLFFT_NOTIMPLEMENTED: return "CLFFT_NOTIMPLEMENTED";
      case CLFFT_TRANSPOSED_NOTIMPLEMENTED: return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
      case CLFFT_FILE_NOT_FOUND: return "CLFFT_FILE_NOT_FOUND";
      case CLFFT_FILE_CREATE_FAILURE: return "CLFFT_FILE_CREATE_FAILURE";
      case CLFFT_VERSION_MISMATCH: return "CLFFT_VERSION_MISMATCH";
      case CLFFT_INVALID_PLAN: return "CLFFT_INVALID_PLAN";
      case CLFFT_DEVICE_NO_DOUBLE: return "CLFFT_DEVICE_NO_DOUBLE";
      case CLFFT_DEVICE_MISMATCH: return "CLFFT_DEVICE_MISMATCH";
      case CLFFT_ENDSTATUS: return "CLFFT_ENDSTATUS";

      default: return "UNKNOWN CL ERROR CODE";
   }
}

void read_cl_file()
{
    FILE *fp;
	// Load the kernel source code into the array source_str
    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(EXIT_FAILURE);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
}  

void cl_load_prog()
{
    //printf("loading OpenCL kernels\n");
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	ASSERT_CL_RETURN(errcode);

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	
    if (errcode == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		
		exit(0);
	}
		
	// Create the OpenCL kernel

    dot_product_and_sum_kernel = clCreateKernel(clProgram, "dot_product_and_sum_kernel", &errcode);

    ASSERT_CL_RETURN(errcode);

	clFinish(clCommandQue);
}
 
void cl_initialization()
{
	errcode = clGetPlatformIDs( 1, &platform_id, NULL );
    errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
    
    errcode = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL );

	errcode = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(driver_version), driver_version,NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(device_version), device_version,NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(device_extension), device_extension,NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
	errcode|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
	errcode|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(global_mem_cacheline_size), &global_mem_cacheline_size, NULL);
    errcode|= clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(global_mem_cache_size), &global_mem_cache_size, NULL);
	errcode|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    //errcode|= clGetDeviceInfo(device_id, CL_DEVICE_PRINTF_BUFFER_SIZE, sizeof(printf_buffer_size), &printf_buffer_size, NULL);
	
    if(errcode == CL_SUCCESS && DEVICE_QUERY == 1){
    printf("Platform found: %s\n", str_temp);
    printf("Device Name: %s\n",str_temp);
    printf("Global Mem Size (MB): %lu\n",global_mem_size/(1024*1024));
    printf("Max Mem Alloc Size Per Mem Object (MB): %ld\n",(long int)max_mem_alloc_size/(1024*1024));
    printf("Device Version: %s\n",device_version);
    printf("Device Extension: %s\n",device_extension);
    printf("OpenCL Driver Version: %s\n",driver_version);
    printf("Local Mem Type (Local=1, Global=2): %d\n",(int)local_mem_type);
    printf("Local Mem Size(KB): %lu\n",local_mem_size/1024);
	printf("Global Mem Size (MB): %lu\n",global_mem_size/(1024*1024));
    printf("Global Mem Cache Size (KB): %d\n",(int)global_mem_cache_size/1024);
	printf("Global Mem Cacheline Size (Bytes): %d\n",(int)global_mem_cacheline_size);
	printf("Max Mem Alloc Size Per Mem Object (MB): %ld\n",(long int)max_mem_alloc_size/(1024*1024));
    //printf("Max Printf Buffer Size (MB): %ld\n",(long int)printf_buffer_size/(1024*1024));
	}
	
    // Create an OpenCL context
    clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
    ASSERT_CL_RETURN(errcode);

    //Create a command-queue
    clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
    ASSERT_CL_RETURN(errcode);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode |= clFinish(clCommandQue);

    errcode |= clReleaseKernel(dot_product_and_sum_kernel);

	errcode |= clReleaseProgram(clProgram);
	errcode |= clReleaseCommandQueue(clCommandQue);
	errcode |= clReleaseContext(clGPUContext);
	ASSERT_CL_RETURN(errcode);
}



