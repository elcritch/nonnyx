import nonnx/private

type
  Env* = ref object
    p: OrtEnv
  SessionOptions* = ref object
    p: OrtSessionOptions
  Session* = ref object
    p: OrtSession
    env: Env
    options: SessionOptions

proc newEnv*(log_severity_level: OrtLoggingLevel, logid: string): Env =
  new(result)
  let api = GetApi()
  let status = api.CreateEnv(log_severity_level, logid, result.p.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtEnv: " & error_message)

proc `$`*(env: Env): string =
  "OrtEnv"

proc newSessionOptions*(): SessionOptions =
  new(result)
  let api = GetApi()
  let status = api.CreateSessionOptions(result.p.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtSessionOptions: " & error_message)

proc enableCpuMemArena*(options: SessionOptions) =
  let api = GetApi()
  let status = api.EnableCpuMemArena(options.p)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to enable CPU memory arena: " & error_message)

proc disableCpuMemArena*(options: SessionOptions) =
  let api = GetApi()
  let status = api.DisableCpuMemArena(options.p)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to disable CPU memory arena: " & error_message)

export private.OrtLoggingLevel, private.ExecutionMode, private.GraphOptimizationLevel

proc newSession*(env: Env, modelPath: string, options: SessionOptions): Session =
  new(result)
  let api = GetApi()
  let status = api.CreateSession(env.p, modelPath, options.p, result.p.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtSession: " & error_message)
  result.env = env
  result.options = options

