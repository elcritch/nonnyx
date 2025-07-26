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

  OrtEnv* = distinct ptr
  OrtStatus* = distinct ptr
  OrtMemoryInfo* = distinct ptr
  OrtIoBinding* = distinct ptr
  OrtSession* = distinct ptr
  OrtValue* = distinct ptr
  OrtRunOptions* = distinct ptr
  OrtTypeInfo* = distinct ptr
  OrtTensorTypeAndShapeInfo* = distinct ptr
  OrtMapTypeInfo* = distinct ptr
  OrtSequenceTypeInfo* = distinct ptr
  OrtOptionalTypeInfo* = distinct ptr
  OrtSessionOptions* = distinct ptr
  OrtCustomOpDomain* = distinct ptr
  OrtModelMetadata* = distinct ptr
  OrtThreadPoolParams* = distinct ptr
  OrtThreadingOptions* = distinct ptr
  OrtArenaCfg* = distinct ptr
  OrtPrepackedWeightsContainer* = distinct ptr
  OrtTensorRTProviderOptionsV2* = distinct ptr
  OrtNvTensorRtRtxProviderOptions* = distinct ptr
  OrtCUDAProviderOptionsV2* = distinct ptr
  OrtCANNProviderOptions* = distinct ptr
  OrtDnnlProviderOptions* = distinct ptr
  OrtOp* = distinct ptr
  OrtOpAttr* = distinct ptr
  OrtLogger* = distinct ptr
  OrtShapeInferContext* = distinct ptr
  OrtLoraAdapter* = distinct ptr
  OrtValueInfo* = distinct ptr
  OrtNode* = distinct ptr
  OrtGraph* = distinct ptr
  OrtModel* = distinct ptr
  OrtModelCompilationOptions* = distinct ptr
  OrtHardwareDevice* = distinct ptr
  OrtEpDevice* = distinct ptr
  OrtKeyValuePairs* = distinct ptr

type
  GraphOptimizationLevel* {.size: sizeof(cint).} = enum
    ORT_DISABLE_ALL = 0, ORT_ENABLE_BASIC = 1, ORT_ENABLE_EXTENDED = 2,
    ORT_ENABLE_ALL = 99

  ExecutionMode* {.size: sizeof(cint).} = enum
    ORT_SEQUENTIAL = 0, ORT_PARALLEL = 1

type
  OrtApi* = object
    CreateStatus*: proc(code: OrtErrorCode, msg: cstring): ptr OrtStatus
    GetErrorCode*: proc(status: ptr OrtStatus): OrtErrorCode
    GetErrorMessage*: proc(status: ptr OrtStatus): cstring
    CreateEnv*: proc(log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr ptr OrtEnv): ptr OrtStatus
    CreateEnvWithCustomLogger*: proc(logging_function: proc(param: pointer, severity: OrtLoggingLevel, category: cstring, logid: cstring, code_location: cstring, message: cstring), logger_param: pointer, log_severity_level: OrtLoggingLevel, logid: cstring, outs: ptr ptr OrtEnv): ptr OrtStatus
    EnableTelemetryEvents*: proc(env: ptr OrtEnv): ptr OrtStatus
    DisableTelemetryEvents*: proc(env: ptr OrtEnv): ptr OrtStatus
    CreateSession*: proc(env: ptr OrtEnv, model_path: cstring, options: ptr OrtSessionOptions, outs: ptr ptr OrtSession): ptr OrtStatus
    CreateSessionFromArray*: proc(env: ptr OrtEnv, model_data: pointer, model_data_length: csize, options: ptr OrtSessionOptions, outs: ptr ptr OrtSession): ptr OrtStatus
    Run*: proc(session: ptr OrtSession, run_options: ptr OrtRunOptions, input_names: ptr cstring, inputs: ptr (ptr OrtValue), input_len: csize, output_names: ptr cstring, output_names_len: csize, outputs: ptr (ptr OrtValue)): ptr OrtStatus
    CreateSessionOptions*: proc(options: ptr ptr OrtSessionOptions): ptr OrtStatus
    SetOptimizedModelFilePath*: proc(options: ptr OrtSessionOptions, optimized_model_filepath: cstring): ptr OrtStatus
    CloneSessionOptions*: proc(in_options: ptr OrtSessionOptions, out_options: ptr ptr OrtSessionOptions): ptr OrtStatus
    SetSessionExecutionMode*: proc(options: ptr OrtSessionOptions, execution_mode: cint): ptr OrtStatus
    EnableProfiling*: proc(options: ptr OrtSessionOptions, profile_file_prefix: cstring): ptr OrtStatus
    DisableProfiling*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    EnableMemPattern*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    DisableMemPattern*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    EnableCpuMemArena*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    DisableCpuMemArena*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    SetSessionLogId*: proc(options: ptr OrtSessionOptions, logid: cstring): ptr OrtStatus
    SetSessionLogVerbosityLevel*: proc(options: ptr OrtSessionOptions, session_log_verbosity_level: cint): ptr OrtStatus
    SetSessionLogSeverityLevel*: proc(options: ptr OrtSessionOptions, session_log_severity_level: cint): ptr OrtStatus
    SetSessionGraphOptimizationLevel*: proc(options: ptr OrtSessionOptions, graph_optimization_level: cint): ptr OrtStatus
    SetIntraOpNumThreads*: proc(options: ptr OrtSessionOptions, intra_op_num_threads: cint): ptr OrtStatus
    SetInterOpNumThreads*: proc(options: ptr OrtSessionOptions, inter_op_num_threads: cint): ptr OrtStatus
    CreateCustomOpDomain*: proc(domain: cstring, outs: ptr ptr OrtCustomOpDomain): ptr OrtStatus
    CustomOpDomain_Add*: proc(custom_op_domain: ptr OrtCustomOpDomain, op: ptr OrtOp): ptr OrtStatus
    AddCustomOpDomain*: proc(options: ptr OrtSessionOptions, custom_op_domain: ptr OrtCustomOpDomain): ptr OrtStatus
    RegisterCustomOpsLibrary*: proc(options: ptr OrtSessionOptions, library_path: cstring, library_handle: ptr pointer): ptr OrtStatus
    SessionGetInputCount*: proc(session: ptr OrtSession, outs: ptr csize): ptr OrtStatus
    SessionGetOutputCount*: proc(session: ptr OrtSession, outs: ptr csize): ptr OrtStatus
    SessionGetOverridableInitializerCount*: proc(session: ptr OrtSession, outs: ptr csize): ptr OrtStatus
    SessionGetInputTypeInfo*: proc(session: ptr OrtSession, index: csize, type_info: ptr ptr OrtTypeInfo): ptr OrtStatus
    SessionGetOutputTypeInfo*: proc(session: ptr OrtSession, index: csize, type_info: ptr ptr OrtTypeInfo): ptr OrtStatus
    SessionGetOverridableInitializerTypeInfo*: proc(session: ptr OrtSession, index: csize, type_info: ptr ptr OrtTypeInfo): ptr OrtStatus
    SessionGetInputName*: proc(session: ptr OrtSession, index: csize, allocator: ptr OrtAllocator, outs: ptr cstring): ptr OrtStatus
    SessionGetOutputName*: proc(session: ptr OrtSession, index: csize, allocator: ptr OrtAllocator, outs: ptr cstring): ptr OrtStatus
    SessionGetOverridableInitializerName*: proc(session: ptr OrtSession, index: csize, allocator: ptr OrtAllocator, outs: ptr cstring): ptr OrtStatus
    CreateRunOptions*: proc(outs: ptr ptr OrtRunOptions): ptr OrtStatus
    RunOptionsSetRunLogVerbosityLevel*: proc(options: ptr OrtRunOptions, log_verbosity_level: cint): ptr OrtStatus
    RunOptionsSetRunLogSeverityLevel*: proc(options: ptr OrtRunOptions, log_severity_level: cint): ptr OrtStatus
    RunOptionsSetRunTag*: proc(options: ptr OrtRunOptions, run_tag: cstring): ptr OrtStatus
    RunOptionsGetRunLogVerbosityLevel*: proc(options: ptr OrtRunOptions, log_verbosity_level: ptr cint): ptr OrtStatus
    RunOptionsGetRunLogSeverityLevel*: proc(options: ptr OrtRunOptions, log_severity_level: ptr cint): ptr OrtStatus
    RunOptionsGetRunTag*: proc(options: ptr OrtRunOptions, run_tag: ptr cstring): ptr OrtStatus
    RunOptionsSetTerminate*: proc(options: ptr OrtRunOptions): ptr OrtStatus
    RunOptionsUnsetTerminate*: proc(options: ptr OrtRunOptions): ptr OrtStatus
    CreateTensorAsOrtValue*: proc(allocator: ptr OrtAllocator, shape: ptr int64, shape_len: csize, `type`: ONNXTensorElementDataType, outs: ptr ptr OrtValue): ptr OrtStatus
    CreateTensorWithDataAsOrtValue*: proc(info: ptr OrtMemoryInfo, p_data: pointer, p_data_len: csize, shape: ptr int64, shape_len: csize, `type`: ONNXTensorElementDataType, outs: ptr ptr OrtValue): ptr OrtStatus
    IsTensor*: proc(value: ptr OrtValue, outs: ptr cint): ptr OrtStatus
    GetTensorMutableData*: proc(value: ptr OrtValue, outs: ptr pointer): ptr OrtStatus
    FillStringTensor*: proc(value: ptr OrtValue, s: ptr cstring, s_len: csize): ptr OrtStatus
    GetStringTensorDataLength*: proc(value: ptr OrtValue, len: ptr csize): ptr OrtStatus
    GetStringTensorContent*: proc(value: ptr OrtValue, s: pointer, s_len: csize, offsets: ptr csize, offsets_len: csize): ptr OrtStatus
    CastTypeInfoToTensorInfo*: proc(type_info: ptr OrtTypeInfo, outs: ptr (ptr OrtTensorTypeAndShapeInfo)): ptr OrtStatus
    GetOnnxTypeFromTypeInfo*: proc(type_info: ptr OrtTypeInfo, outs: ptr ONNXType): ptr OrtStatus
    CreateTensorTypeAndShapeInfo*: proc(outs: ptr ptr OrtTensorTypeAndShapeInfo): ptr OrtStatus
    SetTensorElementType*: proc(info: ptr OrtTensorTypeAndShapeInfo, `type`: ONNXTensorElementDataType): ptr OrtStatus
    SetDimensions*: proc(info: ptr OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_count: csize): ptr OrtStatus
    GetTensorElementType*: proc(info: ptr OrtTensorTypeAndShapeInfo, outs: ptr ONNXTensorElementDataType): ptr OrtStatus
    GetDimensionsCount*: proc(info: ptr OrtTensorTypeAndShapeInfo, outs: ptr csize): ptr OrtStatus
    GetDimensions*: proc(info: ptr OrtTensorTypeAndShapeInfo, dim_values: ptr int64, dim_values_length: csize): ptr OrtStatus
    GetSymbolicDimensions*: proc(info: ptr OrtTensorTypeAndShapeInfo, dim_params: ptr cstring, dim_params_length: csize): ptr OrtStatus
    GetTensorShapeElementCount*: proc(info: ptr OrtTensorTypeAndShapeInfo, outs: ptr csize): ptr OrtStatus
    GetTensorTypeAndShape*: proc(value: ptr OrtValue, outs: ptr ptr OrtTensorTypeAndShapeInfo): ptr OrtStatus
    GetTypeInfo*: proc(value: ptr OrtValue, outs: ptr ptr OrtTypeInfo): ptr OrtStatus
    GetValueType*: proc(value: ptr OrtValue, outs: ptr ONNXType): ptr OrtStatus
    CreateMemoryInfo*: proc(name: cstring, `type`: cint, id: cint, mem_type: cint, outs: ptr ptr OrtMemoryInfo): ptr OrtStatus
    CreateCpuMemoryInfo*: proc(`type`: cint, mem_type: cint, outs: ptr ptr OrtMemoryInfo): ptr OrtStatus
    CompareMemoryInfo*: proc(info1: ptr OrtMemoryInfo, info2: ptr OrtMemoryInfo, outs: ptr cint): ptr OrtStatus
    MemoryInfoGetName*: proc(info: ptr OrtMemoryInfo, outs: ptr cstring): ptr OrtStatus
    MemoryInfoGetId*: proc(info: ptr OrtMemoryInfo, outs: ptr cint): ptr OrtStatus
    MemoryInfoGetMemType*: proc(info: ptr OrtMemoryInfo, outs: ptr cint): ptr OrtStatus
    MemoryInfoGetType*: proc(info: ptr OrtMemoryInfo, outs: ptr cint): ptr OrtStatus
    AllocatorAlloc*: proc(ort_allocator: ptr OrtAllocator, size: csize, outs: ptr pointer): ptr OrtStatus
    AllocatorFree*: proc(ort_allocator: ptr OrtAllocator, p: pointer): ptr OrtStatus
    AllocatorGetInfo*: proc(ort_allocator: ptr OrtAllocator, outs: ptr (ptr OrtMemoryInfo)): ptr OrtStatus
    GetAllocatorWithDefaultOptions*: proc(outs: ptr ptr OrtAllocator): ptr OrtStatus
    AddFreeDimensionOverride*: proc(options: ptr OrtSessionOptions, dim_denotation: cstring, dim_value: int64): ptr OrtStatus
    GetValue*: proc(value: ptr OrtValue, index: cint, allocator: ptr OrtAllocator, outs: ptr ptr OrtValue): ptr OrtStatus
    GetValueCount*: proc(value: ptr OrtValue, outs: ptr csize): ptr OrtStatus
    CreateValue*: proc(in_val: ptr (ptr OrtValue), num_values: csize, value_type: ONNXType, outs: ptr ptr OrtValue): ptr OrtStatus
    CreateOpaqueValue*: proc(domain_name: cstring, type_name: cstring, data_container: pointer, data_container_size: csize, outs: ptr ptr OrtValue): ptr OrtStatus
    GetOpaqueValue*: proc(domain_name: cstring, type_name: cstring, in_val: ptr OrtValue, data_container: pointer, data_container_size: csize): ptr OrtStatus
    KernelInfoGetAttribute_float*: proc(info: ptr OrtOp, name: cstring, outs: ptr float): ptr OrtStatus
    KernelInfoGetAttribute_int64*: proc(info: ptr OrtOp, name: cstring, outs: ptr int64): ptr OrtStatus
    KernelInfoGetAttribute_string*: proc(info: ptr OrtOp, name: cstring, outs: cstring, size: ptr csize): ptr OrtStatus
    KernelContext_GetInputCount*: proc(context: ptr OrtOp, outs: ptr csize): ptr OrtStatus
    KernelContext_GetOutputCount*: proc(context: ptr OrtOp, outs: ptr csize): ptr OrtStatus
    KernelContext_GetInput*: proc(context: ptr OrtOp, index: csize, outs: ptr (ptr OrtValue)): ptr OrtStatus
    KernelContext_GetOutput*: proc(context: ptr OrtOp, index: csize, dim_values: ptr int64, dim_count: csize, outs: ptr ptr OrtValue): ptr OrtStatus
    ReleaseEnv*: proc(input: ptr OrtEnv)
    ReleaseStatus*: proc(input: ptr OrtStatus)
    ReleaseMemoryInfo*: proc(input: ptr OrtMemoryInfo)
    ReleaseSession*: proc(input: ptr OrtSession)
    ReleaseValue*: proc(input: ptr OrtValue)
    ReleaseRunOptions*: proc(input: ptr OrtRunOptions)
    ReleaseTypeInfo*: proc(input: ptr OrtTypeInfo)
    ReleaseTensorTypeAndShapeInfo*: proc(input: ptr OrtTensorTypeAndShapeInfo)
    ReleaseSessionOptions*: proc(input: ptr OrtSessionOptions)
    ReleaseCustomOpDomain*: proc(input: ptr OrtCustomOpDomain)
    ReleaseMapTypeInfo*: proc(input: ptr OrtMapTypeInfo)
    ReleaseSequenceTypeInfo*: proc(input: ptr OrtSequenceTypeInfo)
    SessionEndProfiling*: proc(session: ptr OrtSession, allocator: ptr OrtAllocator, outs: ptr cstring): ptr OrtStatus
    SessionGetModelMetadata*: proc(session: ptr OrtSession, outs: ptr ptr OrtModelMetadata): ptr OrtStatus
    ModelMetadataGetProducerName*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): ptr OrtStatus
    ModelMetadataGetGraphName*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): ptr OrtStatus
    ModelMetadataGetDomain*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): ptr OrtStatus
    ModelMetadataGetDescription*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, value: ptr cstring): ptr OrtStatus
    ModelMetadataLookupCustomMetadataMap*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, key: cstring, value: ptr cstring): ptr OrtStatus
    ModelMetadataGetVersion*: proc(model_metadata: ptr OrtModelMetadata, value: ptr int64): ptr OrtStatus
    ReleaseModelMetadata*: proc(input: ptr OrtModelMetadata)
    CreateEnvWithGlobalThreadPools*: proc(log_severity_level: OrtLoggingLevel, logid: cstring, tp_options: ptr OrtThreadingOptions, outs: ptr ptr OrtEnv): ptr OrtStatus
    DisablePerSessionThreads*: proc(options: ptr OrtSessionOptions): ptr OrtStatus
    CreateThreadingOptions*: proc(outs: ptr ptr OrtThreadingOptions): ptr OrtStatus
    ReleaseThreadingOptions*: proc(input: ptr OrtThreadingOptions)
    ModelMetadataGetCustomMetadataMapKeys*: proc(model_metadata: ptr OrtModelMetadata, allocator: ptr OrtAllocator, keys: ptr (ptr cstring), num_keys: ptr int64): ptr OrtStatus
    AddFreeDimensionOverrideByName*: proc(options: ptr OrtSessionOptions, dim_name: cstring, dim_value: int64): ptr OrtStatus
    GetAvailableProviders*: proc(outs: ptr (ptr cstring), provider_length: ptr cint): ptr OrtStatus
    ReleaseAvailableProviders*: proc(input: ptr cstring, providers_length: cint): ptr OrtStatus
    GetStringTensorElementLength*: proc(value: ptr OrtValue, index: csize, outs: ptr csize): ptr OrtStatus
    GetStringTensorElement*: proc(value: ptr OrtValue, s_len: csize, index: csize, s: pointer): ptr OrtStatus
    FillStringTensorElement*: proc(value: ptr OrtValue, s: cstring, index: csize): ptr OrtStatus
    AddSessionConfigEntry*: proc(options: ptr OrtSessionOptions, config_key: cstring, config_value: cstring): ptr OrtStatus
    CreateAllocator*: proc(session: ptr OrtSession, mem_info: ptr OrtMemoryInfo, outs: ptr ptr OrtAllocator): ptr OrtStatus
    ReleaseAllocator*: proc(input: ptr OrtAllocator)
    RunWithBinding*: proc(session: ptr OrtSession, run_options: ptr OrtRunOptions, binding_ptr: ptr OrtIoBinding): ptr OrtStatus
    CreateIoBinding*: proc(session: ptr OrtSession, outs: ptr ptr OrtIoBinding): ptr OrtStatus
    ReleaseIoBinding*: proc(input: ptr OrtIoBinding)
    BindInput*: proc(binding_ptr: ptr OrtIoBinding, name: cstring, val_ptr: ptr OrtValue): ptr OrtStatus
    BindOutput*: proc(binding_ptr: ptr OrtIoBinding, name: cstring, val_ptr: ptr OrtValue): ptr OrtStatus
    BindOutputToDevice*: proc(binding_ptr: ptr OrtIoBinding, name: cstring, mem_info_ptr: ptr OrtMemoryInfo): ptr OrtStatus
    GetBoundOutputNames*: proc(binding_ptr: ptr OrtIoBinding, allocator: ptr OrtAllocator, buffer: ptr cstring, lengths: ptr (ptr csize), count: ptr csize): ptr OrtStatus
    GetBoundOutputValues*: proc(binding_ptr: ptr OrtIoBinding, allocator: ptr OrtAllocator, output: ptr (ptr (ptr OrtValue)), output_count: ptr csize): ptr OrtStatus
    ClearBoundInputs*: proc(binding_ptr: ptr OrtIoBinding)
    ClearBoundOutputs*: proc(binding_ptr: ptr OrtIoBinding)
    TensorAt*: proc(value: ptr OrtValue, location_values: ptr int64, location_values_count: csize, outs: ptr pointer): ptr OrtStatus
    CreateAndRegisterAllocator*: proc(env: ptr OrtEnv, mem_info: ptr OrtMemoryInfo, arena_cfg: ptr OrtArenaCfg): ptr OrtStatus
    SetLanguageProjection*: proc(ort_env: ptr OrtEnv, projection: cint): ptr OrtStatus
    SessionGetProfilingStartTimeNs*: proc(session: ptr OrtSession, outs: ptr uint64): ptr OrtStatus
    SetGlobalIntraOpNumThreads*: proc(tp_options: ptr OrtThreadingOptions, intra_op_num_threads: cint): ptr OrtStatus
    SetGlobalInterOpNumThreads*: proc(tp_options: ptr OrtThreadingOptions, inter_op_num_threads: cint): ptr OrtStatus

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