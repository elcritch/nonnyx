import nonnx_private

type
  Env* = object
    p: ptr OrtEnv

proc newEnv*(log_severity_level: OrtLoggingLevel, logid: string): Env =
  var env: ptr OrtEnv
  let api = GetApi()
  let status = api.CreateEnv(log_severity_level, logid, env.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtEnv: " & error_message)
  result = Env(p: env)

proc `$`*(env: Env): string =
  "OrtEnv"

proc `=destroy`*(env: Env) =
  if env.p != nil:
    let api = GetApi()
    api.ReleaseEnv(env.p)

type
  SessionOptions* = object
    p: ptr OrtSessionOptions

proc newSessionOptions*(): SessionOptions =
  var options: ptr OrtSessionOptions
  let api = GetApi()
  let status = api.CreateSessionOptions(options.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtSessionOptions: " & error_message)
  result = SessionOptions(p: options)

proc `=destroy`*(options: SessionOptions) =
  if options.p != nil:
    let api = GetApi()
    api.ReleaseSessionOptions(options.p)

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

type
  Session* = object
    p: ptr OrtSession
    env: Env
    options: SessionOptions

export nonnx_private.OrtLoggingLevel, nonnx_private.ExecutionMode, nonnx_private.GraphOptimizationLevel

proc newSession*(env: Env, modelPath: string, options: SessionOptions): Session =
  var session: ptr OrtSession
  let api = GetApi()
  let status = api.CreateSession(env.p, modelPath, options.p, session.addr)
  if status != nil:
    let error_message = $api.GetErrorMessage(status)
    api.ReleaseStatus(status)
    raise newException(Exception, "Failed to create OrtSession: " & error_message)
  result = Session(p: session, env: env, options: options)

proc `=destroy`*(session: Session) =
  if session.p != nil:
    let api = GetApi()
    api.ReleaseSession(session.p)
