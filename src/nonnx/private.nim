const
  ORT_API_VERSION* = 15

{.push, header: "onnxruntime_c_api.h", cdecl.}

type
  ONNXTensorElementDataType* {.size: sizeof(cint).} = enum
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   ## maps to c type float
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,   ## maps to c type uint8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,    ## maps to c type int8_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,  ## maps to c type uint16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,   ## maps to c type int16_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,   ## maps to c type int32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,   ## maps to c type int64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,  ## maps to c++ type std::string
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,         ## maps to c type double
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,         ## maps to c type uint32_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,         ## maps to c type uint64_t
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,      ## complex with float32 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,     ## complex with float64 real and imaginary components
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,       ## Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,   ## Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ, ## Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,     ## Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ, ## Non-IEEE floating-point format based on IEEE754 single-precision
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,          ## maps to a pair of packed uint4 values (size == 1 byte)
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4            ## maps to a pair of packed int4 values (size == 1 byte)

type
  ONNXType* {.size: sizeof(cint).} = enum
    ONNX_TYPE_UNKNOWN, ONNX_TYPE_TENSOR, ONNX_TYPE_SEQUENCE, ONNX_TYPE_MAP,
    ONNX_TYPE_OPAQUE, ONNX_TYPE_SPARSETENSOR, ONNX_TYPE_OPTIONAL

  OrtSparseFormat* {.size: sizeof(cint).} = enum
    ORT_SPARSE_UNDEFINED = 0, ORT_SPARSE_COO = 0x1, ORT_SPARSE_CSRC = 0x2,
    ORT_SPARSE_BLOCK_SPARSE = 0x4

  OrtSparseIndicesFormat* {.size: sizeof(cint).} = enum
    ORT_SPARSE_COO_INDICES, ORT_SPARSE_CSR_INNER_INDICES,
    ORT_SPARSE_CSR_OUTER_INDICES, ORT_SPARSE_BLOCK_SPARSE_INDICES

  OrtLoggingLevel* {.size: sizeof(cint).} = enum
    ORT_LOGGING_LEVEL_VERBOSE,
    ORT_LOGGING_LEVEL_INFO,
    ORT_LOGGING_LEVEL_WARNING,
    ORT_LOGGING_LEVEL_ERROR,
    ORT_LOGGING_LEVEL_FATAL

  OrtErrorCode* {.size: sizeof(cint).} = enum
    ORT_OK, ORT_FAIL, ORT_INVALID_ARGUMENT, ORT_NO_SUCHFILE, ORT_NO_MODEL,
    ORT_ENGINE_ERROR, ORT_RUNTIME_EXCEPTION, ORT_INVALID_PROTOBUF,
    ORT_MODEL_LOADED, ORT_NOT_IMPLEMENTED, ORT_INVALID_GRAPH, ORT_EP_FAIL,
    ORT_MODEL_LOAD_CANCELED, ORT_MODEL_REQUIRES_COMPILATION

  OrtOpAttrType* {.size: sizeof(cint).} = enum
    ORT_OP_ATTR_UNDEFINED = 0, ORT_OP_ATTR_INT, ORT_OP_ATTR_INTS,
    ORT_OP_ATTR_FLOAT, ORT_OP_ATTR_FLOATS, ORT_OP_ATTR_STRING,
    ORT_OP_ATTR_STRINGS

  OrtEnvObj = object
  OrtEnv* = ptr OrtEnvObj
  OrtStatusObj = object
  OrtStatus* = ptr OrtStatusObj
  OrtMemoryInfoObj = object
  OrtMemoryInfo* = ptr OrtMemoryInfoObj
  OrtIoBindingObj = object
  OrtIoBinding* = ptr OrtIoBindingObj
  OrtSessionObj = object
  OrtSession* = ptr OrtSessionObj
  OrtValueObj = object
  OrtValue* = ptr OrtValueObj
  OrtRunOptionsObj = object
  OrtRunOptions* = ptr OrtRunOptionsObj
  OrtTypeInfoObj = object
  OrtTypeInfo* = ptr OrtTypeInfoObj
  OrtTensorTypeAndShapeInfoObj = object
  OrtTensorTypeAndShapeInfo* = ptr OrtTensorTypeAndShapeInfoObj
  OrtMapTypeInfoObj = object
  OrtMapTypeInfo* = ptr OrtMapTypeInfoObj
  OrtSequenceTypeInfoObj = object
  OrtSequenceTypeInfo* = ptr OrtSequenceTypeInfoObj
  OrtOptionalTypeInfoObj = object
  OrtOptionalTypeInfo* = ptr OrtOptionalTypeInfoObj
  OrtSessionOptionsObj = object
  OrtSessionOptions* = ptr OrtSessionOptionsObj
  OrtCustomOpDomainObj = object
  OrtCustomOpDomain* = ptr OrtCustomOpDomainObj
  OrtModelMetadataObj = object
  OrtModelMetadata* = ptr OrtModelMetadataObj
  OrtThreadPoolParamsObj = object
  OrtThreadPoolParams* = ptr OrtThreadPoolParamsObj
  OrtThreadingOptionsObj = object
  OrtThreadingOptions* = ptr OrtThreadingOptionsObj
  OrtArenaCfgObj = object
  OrtArenaCfg* = ptr OrtArenaCfgObj
  OrtPrepackedWeightsContainerObj = object
  OrtPrepackedWeightsContainer* = ptr OrtPrepackedWeightsContainerObj
  OrtTensorRTProviderOptionsV2Obj = object
  OrtTensorRTProviderOptionsV2* = ptr OrtTensorRTProviderOptionsV2Obj
  OrtNvTensorRtRtxProviderOptionsObj = object
  OrtNvTensorRtRtxProviderOptions* = ptr OrtNvTensorRtRtxProviderOptionsObj
  OrtCUDAProviderOptionsV2Obj = object
  OrtCUDAProviderOptionsV2* = ptr OrtCUDAProviderOptionsV2Obj
  OrtCANNProviderOptionsObj = object
  OrtCANNProviderOptions* = ptr OrtCANNProviderOptionsObj
  OrtDnnlProviderOptionsObj = object
  OrtDnnlProviderOptions* = ptr OrtDnnlProviderOptionsObj
  OrtOpObj = object
  OrtOp* = ptr OrtOpObj
  OrtOpAttrObj = object
  OrtOpAttr* = ptr OrtOpAttrObj
  OrtLoggerObj = object
  OrtLogger* = ptr OrtLoggerObj
  OrtShapeInferContextObj = object
  OrtShapeInferContext* = ptr OrtShapeInferContextObj
  OrtLoraAdapterObj = object
  OrtLoraAdapter* = ptr OrtLoraAdapterObj
  OrtValueInfoObj = object
  OrtValueInfo* = ptr OrtValueInfoObj
  OrtNodeObj = object
  OrtNode* = ptr OrtNodeObj
  OrtGraphObj = object
  OrtGraph* = ptr OrtGraphObj
  OrtModelObj = object
  OrtModel* = ptr OrtModelObj
  OrtModelCompilationOptionsObj = object
  OrtModelCompilationOptions* = ptr OrtModelCompilationOptionsObj
  OrtHardwareDeviceObj = object
  OrtHardwareDevice* = ptr OrtHardwareDeviceObj
  OrtEpDeviceObj = object
  OrtEpDevice* = ptr OrtEpDeviceObj
  OrtKeyValuePairsObj = object
  OrtKeyValuePairs* = ptr OrtKeyValuePairsObj
  OrtAllocatorObj = object
  OrtAllocator* = ptr OrtAllocatorObj

type
  GraphOptimizationLevel* {.size: sizeof(cint).} = enum
    ORT_DISABLE_ALL = 0, ORT_ENABLE_BASIC = 1, ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99

  ExecutionMode* {.size: sizeof(cint).} = enum
    ORT_SEQUENTIAL = 0, ORT_PARALLEL = 1

type
  OrtApi* = object
    CreateStatus*: proc(code: OrtErrorCode, msg: cstring): OrtStatus
    GetErrorCode*: proc(status: OrtStatus): OrtErrorCode
    GetErrorMessage*: proc(status: OrtStatus): cstring
    CreateEnv*: proc(log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr OrtEnv): OrtStatus
    CreateEnvWithCustomLogger*: proc(logging_function: proc(param: pointer, severity: OrtLoggingLevel, category: cstring, logid: cstring, code_location: cstring, message: cstring), logger_param: pointer, log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr OrtEnv): OrtStatus
    EnableTelemetryEvents*: proc(env: OrtEnv): OrtStatus
    DisableTelemetryEvents*: proc(env: OrtEnv): OrtStatus
    CreateSession*: proc(env: OrtEnv, model_path: cstring, options: OrtSessionOptions, outs: ptr OrtSession): OrtStatus
    CreateSessionFromArray*: proc(env: OrtEnv, model_data: pointer, model_data_length: csize_t, options: OrtSessionOptions, outs: ptr OrtSession): OrtStatus
    Run*: proc(session: OrtSession, run_options: OrtRunOptions, input_names: ptr cstring, inputs: ptr OrtValue, input_len: csize_t, output_names: ptr cstring, output_names_len: csize_t, outputs: ptr OrtValue): OrtStatus
    CreateSessionOptions*: proc(options: ptr OrtSessionOptions): OrtStatus
    SetOptimizedModelFilePath*: proc(options: OrtSessionOptions, optimized_model_filepath: cstring): OrtStatus
    CloneSessionOptions*: proc(in_options: OrtSessionOptions, out_options: ptr OrtSessionOptions): OrtStatus
    SetSessionExecutionMode*: proc(options: OrtSessionOptions, execution_mode: cint): OrtStatus
    EnableProfiling*: proc(options: OrtSessionOptions, profile_file_prefix: cstring): OrtStatus
    DisableProfiling*: proc(options: OrtSessionOptions): OrtStatus
    EnableMemPattern*: proc(options: OrtSessionOptions): OrtStatus
    DisableMemPattern*: proc(options: OrtSessionOptions): OrtStatus
    EnableCpuMemArena*: proc(options: OrtSessionOptions): OrtStatus
    DisableCpuMemArena*: proc(options: OrtSessionOptions): OrtStatus
    SetSessionLogId*: proc(options: OrtSessionOptions, logid: cstring): OrtStatus
    SetSessionLogVerbosityLevel*: proc(options: OrtSessionOptions, session_log_verbosity_level: cint): OrtStatus
    SetSessionLogSeverityLevel*: proc(options: OrtSessionOptions, session_log_severity_level: cint): OrtStatus
    SetSessionGraphOptimizationLevel*: proc(options: OrtSessionOptions, graph_optimization_level: cint): OrtStatus
    SetIntraOpNumThreads*: proc(options: OrtSessionOptions, intra_op_num_threads: cint): OrtStatus
    SetInterOpNumThreads*: proc(options: OrtSessionOptions, inter_op_num_threads: cint): OrtStatus
    CreateCustomOpDomain*: proc(domain: cstring, outs: ptr OrtCustomOpDomain): OrtStatus
    CustomOpDomain_Add*: proc(custom_op_domain: OrtCustomOpDomain, op: OrtOp): OrtStatus
    AddCustomOpDomain*: proc(options: OrtSessionOptions, custom_op_domain: OrtCustomOpDomain): OrtStatus
    RegisterCustomOpsLibrary*: proc(options: OrtSessionOptions, library_path: cstring, library_handle: ptr pointer): OrtStatus
    SessionGetInputCount*: proc(session: OrtSession, outs: ptr csize_t): OrtStatus
    SessionGetOutputCount*: proc(session: OrtSession, outs: ptr csize_t): OrtStatus
    SessionGetOverridableInitializerCount*: proc(session: OrtSession, outs: ptr csize_t): OrtStatus
    SessionGetInputTypeInfo*: proc(session: OrtSession, index: csize_t, type_info: ptr OrtTypeInfo): OrtStatus
    SessionGetOutputTypeInfo*: proc(session: OrtSession, index: csize_t, type_info: ptr OrtTypeInfo): OrtStatus
    SessionGetOverridableInitializerTypeInfo*: proc(session: OrtSession, index: csize_t, type_info: ptr OrtTypeInfo): OrtStatus
    SessionGetInputName*: proc(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatus
    SessionGetOutputName*: proc(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatus
    SessionGetOverridableInitializerName*: proc(session: OrtSession, index: csize_t, allocator: OrtAllocator, outs: ptr cstring): OrtStatus
    CreateRunOptions*: proc(outs: ptr OrtRunOptions): OrtStatus
    RunOptionsSetRunLogVerbosityLevel*: proc(options: OrtRunOptions, log_verbosity_level: cint): OrtStatus
    RunOptionsSetRunLogSeverityLevel*: proc(options: OrtRunOptions, log_severity_level: cint): OrtStatus
    RunOptionsSetRunTag*: proc(options: OrtRunOptions, run_tag: cstring): OrtStatus
    RunOptionsGetRunLogVerbosityLevel*: proc(options: OrtRunOptions, log_verbosity_level: ptr cint): OrtStatus
    RunOptionsGetRunLogSeverityLevel*: proc(options: OrtRunOptions, log_severity_level: ptr cint): OrtStatus
    RunOptionsGetRunTag*: proc(options: OrtRunOptions, run_tag: ptr cstring): OrtStatus
    RunOptionsSetTerminate*: proc(options: OrtRunOptions): OrtStatus
    RunOptionsUnsetTerminate*: proc(options: OrtRunOptions): OrtStatus
    CreateTensorAsOrtValue*: proc(allocator: OrtAllocator, shape: ptr int64, shape_len: csize_t, `type`: ONNXTensorElementDataType, outs: ptr OrtValue): OrtStatus
    CreateTensorWithDataAsOrtValue*: proc(info: OrtMemoryInfo, p_data: pointer, p_data_len: csize_t, shape: ptr int64, shape_len: csize_t, `type`: ONNXTensorElementDataType, outs: ptr OrtValue): OrtStatus
    IsTensor*: proc(value: OrtValue, outs: ptr cint): OrtStatus
    GetTensorMutableData*: proc(value: OrtValue, outs: ptr pointer): OrtStatus
    FillStringTensor*: proc(value: OrtValue, s: ptr cstring, s_len: csize_t): OrtStatus
    GetStringTensorDataLength*: proc(value: OrtValue, len: ptr csize_t): OrtStatus
    GetStringTensorContent*: proc(value: OrtValue, s: pointer, s_len: csize_t, offsets: ptr csize_t, offsets_len: csize_t): OrtStatus
    CastTypeInfoToTensorInfo*: proc(type_info: OrtTypeInfo, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatus
    GetOnnxTypeFromTypeInfo*: proc(type_info: OrtTypeInfo, outs: ptr ONNXType): OrtStatus
    CreateTensorTypeAndShapeInfo*: proc(outs: ptr OrtTensorTypeAndShapeInfo): OrtStatus
    SetTensorElementType*: proc(info: OrtTensorTypeAndShapeInfo, `type`: ONNXTensorElementDataType): OrtStatus
    SetDimensions*: proc(info: OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_count: csize_t): OrtStatus
    GetTensorElementType*: proc(info: OrtTensorTypeAndShapeInfo, outs: ptr ONNXTensorElementDataType): OrtStatus
    GetDimensionsCount*: proc(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatus
    GetDimensions*: proc(info: OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize_t): OrtStatus
    GetSymbolicDimensions*: proc(info: OrtTensorTypeAndShapeInfo, dim_params: ptr cstring, dim_params_length: csize_t): OrtStatus
    GetTensorShapeElementCount*: proc(info: OrtTensorTypeAndShapeInfo, outs: ptr csize_t): OrtStatus
    GetTensorTypeAndShape*: proc(value: OrtValue, outs: ptr OrtTensorTypeAndShapeInfo): OrtStatus
    GetTypeInfo*: proc(value: OrtValue, outs: ptr OrtTypeInfo): OrtStatus
    GetValueType*: proc(value: OrtValue, outs: ptr ONNXType): OrtStatus
    CreateMemoryInfo*: proc(name: cstring, `type`: cint, id: cint, mem_type: cint, outs: ptr OrtMemoryInfo): OrtStatus
    CreateCpuMemoryInfo*: proc(`type`: cint, mem_type: cint, outs: ptr OrtMemoryInfo): OrtStatus
    CompareMemoryInfo*: proc(info1: OrtMemoryInfo, info2: OrtMemoryInfo, outs: ptr cint): OrtStatus
    MemoryInfoGetName*: proc(info: OrtMemoryInfo, outs: ptr cstring): OrtStatus
    MemoryInfoGetId*: proc(info: OrtMemoryInfo, outs: ptr cint): OrtStatus
    MemoryInfoGetMemType*: proc(info: OrtMemoryInfo, outs: ptr cint): OrtStatus
    MemoryInfoGetType*: proc(info: OrtMemoryInfo, outs: ptr cint): OrtStatus
    AllocatorAlloc*: proc(ort_allocator: OrtAllocator, size: csize_t, outs: ptr pointer): OrtStatus
    AllocatorFree*: proc(ort_allocator: OrtAllocator, p: pointer): OrtStatus
    AllocatorGetInfo*: proc(ort_allocator: OrtAllocator, outs: ptr OrtMemoryInfo): OrtStatus
    GetAllocatorWithDefaultOptions*: proc(outs: ptr OrtAllocator): OrtStatus
    AddFreeDimensionOverride*: proc(options: OrtSessionOptions, dim_denotation: cstring, dim_value: int64): OrtStatus
    GetValue*: proc(value: OrtValue, index: cint, allocator: OrtAllocator, outs: ptr OrtValue): OrtStatus
    GetValueCount*: proc(value: OrtValue, outs: ptr csize_t): OrtStatus
    CreateValue*: proc(in_val: ptr OrtValue, num_values: csize_t, value_type: ONNXType, outs: ptr OrtValue): OrtStatus
    CreateOpaqueValue*: proc(domain_name: cstring, type_name: cstring, data_container: pointer, data_container_size: csize_t, outs: ptr OrtValue): OrtStatus
    GetOpaqueValue*: proc(domain_name: cstring, type_name: cstring, in_val: OrtValue, data_container: pointer, data_container_size: csize_t): OrtStatus
    KernelInfoGetAttribute_float*: proc(info: OrtOp, name: cstring, outs: ptr float): OrtStatus
    KernelInfoGetAttribute_int64*: proc(info: OrtOp, name: cstring, outs: ptr int64): OrtStatus
    KernelInfoGetAttribute_string*: proc(info: OrtOp, name: cstring, outs: cstring, size: ptr csize_t): OrtStatus
    KernelContext_GetInputCount*: proc(context: OrtOp, outs: ptr csize_t): OrtStatus
    KernelContext_GetOutputCount*: proc(context: OrtOp, outs: ptr csize_t): OrtStatus
    KernelContext_GetInput*: proc(context: OrtOp, index: csize_t, outs: ptr OrtValue): OrtStatus
    KernelContext_GetOutput*: proc(context: OrtOp, index: csize_t, dim_values: ptr int64, dim_count: csize_t, outs: ptr OrtValue): OrtStatus
    ReleaseEnv*: proc(input: OrtEnv)
    ReleaseStatus*: proc(input: OrtStatus)
    ReleaseMemoryInfo*: proc(input: OrtMemoryInfo)
    ReleaseSession*: proc(input: OrtSession)
    ReleaseValue*: proc(input: OrtValue)
    ReleaseRunOptions*: proc(input: OrtRunOptions)
    ReleaseTypeInfo*: proc(input: OrtTypeInfo)
    ReleaseTensorTypeAndShapeInfo*: proc(input: OrtTensorTypeAndShapeInfo)
    ReleaseSessionOptions*: proc(input: OrtSessionOptions)
    ReleaseCustomOpDomain*: proc(input: OrtCustomOpDomain)
    ReleaseMapTypeInfo*: proc(input: OrtMapTypeInfo)
    ReleaseSequenceTypeInfo*: proc(input: OrtSequenceTypeInfo)
    SessionEndProfiling*: proc(session: OrtSession, allocator: OrtAllocator, outs: ptr cstring): OrtStatus
    SessionGetModelMetadata*: proc(session: OrtSession, outs: ptr OrtModelMetadata): OrtStatus
    ModelMetadataGetProducerName*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, value: ptr cstring): OrtStatus
    ModelMetadataGetGraphName*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, value: ptr cstring): OrtStatus
    ModelMetadataGetDomain*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, value: ptr cstring): OrtStatus
    ModelMetadataGetDescription*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, value: ptr cstring): OrtStatus
    ModelMetadataLookupCustomMetadataMap*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, key: cstring, value: ptr cstring): OrtStatus
    ModelMetadataGetVersion*: proc(model_metadata: OrtModelMetadata, value: ptr int64): OrtStatus
    ReleaseModelMetadata*: proc(input: OrtModelMetadata)
    CreateEnvWithGlobalThreadPools*: proc(log_severity_level: OrtLoggingLevel, logid: cstring, tp_options: OrtThreadingOptions, outs: ptr OrtEnv): OrtStatus
    DisablePerSessionThreads*: proc(options: OrtSessionOptions): OrtStatus
    CreateThreadingOptions*: proc(outs: ptr OrtThreadingOptions): OrtStatus
    ReleaseThreadingOptions*: proc(input: OrtThreadingOptions)
    ModelMetadataGetCustomMetadataMapKeys*: proc(model_metadata: OrtModelMetadata, allocator: OrtAllocator, keys: ptr (ptr cstring), num_keys: ptr int64): OrtStatus
    AddFreeDimensionOverrideByName*: proc(options: OrtSessionOptions, dim_name: cstring, dim_value: int64): OrtStatus
    GetAvailableProviders*: proc(outs: ptr (ptr cstring), provider_length: ptr cint): OrtStatus
    ReleaseAvailableProviders*: proc(input: ptr cstring, providers_length: cint): OrtStatus
    GetStringTensorElementLength*: proc(value: OrtValue, index: csize_t, outs: ptr csize_t): OrtStatus
    GetStringTensorElement*: proc(value: OrtValue, s_len: csize_t, index: csize_t, s: pointer): OrtStatus
    FillStringTensorElement*: proc(value: OrtValue, s: cstring, index: csize_t): OrtStatus
    AddSessionConfigEntry*: proc(options: OrtSessionOptions, config_key: cstring, config_value: cstring): OrtStatus
    CreateAllocator*: proc(session: OrtSession, mem_info: OrtMemoryInfo, outs: ptr OrtAllocator): OrtStatus
    ReleaseAllocator*: proc(input: OrtAllocator)
    RunWithBinding*: proc(session: OrtSession, run_options: OrtRunOptions, binding_ptr: OrtIoBinding): OrtStatus
    CreateIoBinding*: proc(session: OrtSession, outs: ptr OrtIoBinding): OrtStatus
    ReleaseIoBinding*: proc(input: OrtIoBinding)
    BindInput*: proc(binding_ptr: OrtIoBinding, name: cstring, val_ptr: OrtValue): OrtStatus
    BindOutput*: proc(binding_ptr: OrtIoBinding, name: cstring, val_ptr: OrtValue): OrtStatus
    BindOutputToDevice*: proc(binding_ptr: OrtIoBinding, name: cstring, mem_info_ptr: OrtMemoryInfo): OrtStatus
    GetBoundOutputNames*: proc(binding_ptr: OrtIoBinding, allocator: OrtAllocator, buffer: ptr cstring, lengths: ptr (ptr csize_t), count: ptr csize_t): OrtStatus
    GetBoundOutputValues*: proc(binding_ptr: OrtIoBinding, allocator: OrtAllocator, output: ptr (ptr OrtValue), output_count: ptr csize_t): OrtStatus
    ClearBoundInputs*: proc(binding_ptr: OrtIoBinding)
    ClearBoundOutputs*: proc(binding_ptr: OrtIoBinding)
    TensorAt*: proc(value: OrtValue, location_values: ptr int64, location_values_count: csize_t, outs: ptr pointer): OrtStatus
    CreateAndRegisterAllocator*: proc(env: OrtEnv, mem_info: OrtMemoryInfo, arena_cfg: OrtArenaCfg): OrtStatus
    SetLanguageProjection*: proc(ort_env: OrtEnv, projection: cint): OrtStatus
    SessionGetProfilingStartTimeNs*: proc(session: OrtSession, outs: ptr uint64): OrtStatus
    SetGlobalIntraOpNumThreads*: proc(tp_options: OrtThreadingOptions, intra_op_num_threads: cint): OrtStatus
    SetGlobalInterOpNumThreads*: proc(tp_options: OrtThreadingOptions, inter_op_num_threads: cint): OrtStatus

type
  OrtApiBase* = object
    GetApi*: proc(version: uint32): ptr OrtApi
    GetVersionString*: proc(): cstring

proc OrtGetApiBase*(): ptr OrtApiBase {.importc: "OrtGetApiBase", dynlib: "libonnxruntime.so".}

var g_ort_api*: ptr OrtApi

proc GetApi*(): ptr OrtApi =
  if g_ort_api.isNil:
    let base = OrtGetApiBase()
    if base.isNil:
      raise newException(Exception, "OrtGetApiBase failed")
    g_ort_api = base.GetApi(ORT_API_VERSION)
    if g_ort_api.isNil:
      raise newException(Exception, "GetApi failed")
  return g_ort_api

{.pop.} 