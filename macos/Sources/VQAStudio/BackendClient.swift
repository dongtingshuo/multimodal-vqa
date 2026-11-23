import Foundation
import Darwin

/// Pipe access is serialized; the process reference and cancellation generation are locked.
final class BackendClient: @unchecked Sendable {
    private let queue = DispatchQueue(label: "studio.vqa.inference", qos: .userInitiated)
    private var process: Process?
    // Connection cleanup can retire a worker before its termination escalation fires.
    // Keep ownership until it actually exits, including after malformed/closed output.
    private var ownedProcesses: [Process] = []
    private var input: FileHandle?
    private var output: FileHandle?
    private var log: FileHandle?
    private let processLock = NSLock()
    private var generation = 0
    private let logDirectory: URL
    private let requestTimeout: TimeInterval

    init(logDirectory: URL = StudioStore.supportURL, requestTimeout: TimeInterval = 180) {
        self.logDirectory = logDirectory
        self.requestTimeout = requestTimeout
    }

    func request(settings: AppSettings, action: String, image: String? = nil, question: String? = nil) async throws -> WorkerResponse {
        try Task.checkCancellation()
        let token = processLock.withLock { generation }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    try checkCancellation(token)
                    if processLock.withLock({ process?.isRunning }) != true { try launch(settings) }
                    try checkCancellation(token)
                    var payload: [String: Any] = ["id": UUID().uuidString, "action": action, "settings": settings.workerSettings, "topk": settings.topk]
                    payload["image"] = image
                    payload["question"] = question
                    var data = try JSONSerialization.data(withJSONObject: payload)
                    data.append(10)
                    guard let input, let output, let running = processLock.withLock({ process }) else { throw failure("模型进程未启动。") }
                    let deadline = RequestDeadline()
                    let timeout = Task.detached { [requestTimeout] in
                        do { try await Task.sleep(for: .seconds(requestTimeout)) }
                        catch { return }
                        if deadline.expire() { Self.stop(running) }
                    }
                    defer { _ = deadline.finish(); timeout.cancel() }
                    try input.write(contentsOf: data)
                    var line = Data()
                    var complete = false
                    while let byte = try output.read(upToCount: 1), !byte.isEmpty {
                        if byte.first == 10 { complete = true; break }
                        line.append(byte)
                        if line.count > 1_048_576 { throw failure("模型返回的数据过大。") }
                    }
                    try checkCancellation(token)
                    if deadline.finish() { throw failure("模型响应超时，已停止当前进程。请重新加载模型；详细信息可从设置打开运行日志。") }
                    guard complete, !line.isEmpty else { throw failure("模型进程意外退出，未收到完整结果。请重新加载模型；详细信息可从设置打开运行日志。") }
                    let response: WorkerResponse
                    do { response = try JSONDecoder().decode(WorkerResponse.self, from: line) }
                    catch { throw failure("模型返回的结果无法读取，请重新加载模型。") }
                    guard response.id == payload["id"] as? String else { throw failure("模型返回了不匹配的结果，请重新加载模型。") }
                    guard response.ok else { throw BackendError.rejected(response.error ?? "模型未能完成请求。") }
                    if action == "predict" {
                        guard let answers = response.answers, !answers.isEmpty, answers.count <= 10,
                              Set(answers.map(\.answer)).count == answers.count,
                              answers.allSatisfy({ !$0.answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.score.isFinite && (0...1).contains($0.score) }),
                              let elapsed = response.elapsed, elapsed.isFinite, elapsed >= 0 else {
                            throw failure("模型返回了无效的候选答案，请重新加载模型。")
                        }
                    }
                    continuation.resume(returning: response)
                } catch {
                    if processLock.withLock({ generation != token }) {
                        cleanup()
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    if (error as? BackendError)?.needsReload != false { cleanup() }
                    continuation.resume(throwing: error)
                }
            }
            }
        } onCancel: {
            self.cancel()
        }
    }

    private func launch(_ settings: AppSettings) throws {
        cleanup()
        guard FileManager.default.isExecutableFile(atPath: settings.python) else { throw failure("找不到 Python，请在设置中选择项目环境的 Python 可执行文件。") }
        guard FileManager.default.fileExists(atPath: URL(fileURLWithPath: settings.project).appendingPathComponent("vqa_project/desktop_worker.py").path) else { throw failure("项目目录无效，请在设置中选择包含 vqa_project 的文件夹。") }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: settings.python)
        // Python -m resolves its initial cwd before executing any module. A cloud-provider
        // getcwd/open stall can block that initialization. Start in local support storage,
        // then select the project explicitly before importing the unchanged worker.
        task.arguments = ["-u", "-c", "import os, sys; project = sys.argv.pop(1); sys.path[0] = project; os.chdir(project); import runpy; runpy.run_module('vqa_project.desktop_worker', run_name='__main__', alter_sys=True)", settings.project]
        task.currentDirectoryURL = logDirectory
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["MPLCONFIGDIR"] = logDirectory.appendingPathComponent("Cache").path
        task.environment = environment
        let stdinPipe = Pipe(), stdoutPipe = Pipe()
        task.standardInput = stdinPipe
        task.standardOutput = stdoutPipe
        let logURL = logDirectory.appendingPathComponent("engine.log")
        try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: logURL.path) { FileManager.default.createFile(atPath: logURL.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        task.standardError = handle
        try task.run()
        processLock.withLock {
            ownedProcesses.removeAll { !$0.isRunning }
            ownedProcesses.append(task)
            process = task
        }
        input = stdinPipe.fileHandleForWriting; output = stdoutPipe.fileHandleForReading; log = handle
    }

    func cancel() {
        let running = processLock.withLock { generation += 1; return ownedProcesses }
        for child in running { Self.stop(child) }
    }

    func shutdown() {
        cancel()
        queue.async { [self] in cleanup() }
    }

    /// Successful App termination and explicit reload must finish ownership, not just send SIGTERM.
    /// Drain pipe cleanup first; a separately awaited task makes exit waiting immune to caller cancellation.
    func shutdownAndWait() async {
        cancel()
        let children: [Process] = await withCheckedContinuation { continuation in
            queue.async { [self] in
                cleanup()
                continuation.resume(returning: processLock.withLock { ownedProcesses })
            }
        }
        // waitUntilExit on the pipe queue intermittently stalled lifecycle stress checks.
        // Observe process exit without blocking that queue in a run loop.
        await Task.detached(priority: .utility) {
            for child in children { Self.stop(child) }
            while children.contains(where: { $0.isRunning }) {
                try? await Task.sleep(for: .milliseconds(20))
            }
        }.value
        processLock.withLock { ownedProcesses.removeAll { !$0.isRunning } }
    }

    private func cleanup() {
        try? input?.close()
        let running = processLock.withLock { let value = process; process = nil; return value }
        if let running { Self.stop(running) }
        try? output?.close(); try? log?.close()
        input = nil; output = nil; log = nil
    }

    private func checkCancellation(_ token: Int) throws {
        if processLock.withLock({ generation != token }) { throw CancellationError() }
    }

    private static func stop(_ running: Process) {
        guard running.isRunning else { return }
        running.terminate()
        // Python can be inside native code or ignore SIGTERM; never leave Stop blocked forever.
        Task.detached {
            try? await Task.sleep(for: .milliseconds(750))
            if running.isRunning { _ = kill(running.processIdentifier, SIGKILL) }
        }
    }

    private func failure(_ message: String) -> BackendError { .connection(message) }
}
