import Foundation

struct SettingsValidationClient: Sendable {
    var timeout: Duration = .seconds(8)

    func message(for settings: AppSettings) async throws -> String? {
        try Task.checkCancellation()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vqa-settings-check-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        let process = StudioHelperProcess(executable: executable, workerArgument: SettingsValidationWorker.argument, directory: directory)
        defer { process.release() }
        let request = try JSONEncoder().encode(settings)
        try request.write(to: directory.appendingPathComponent("request.json"), options: .atomic)
        do {
            return try await withTaskCancellationHandler {
                try Task.checkCancellation()
                try process.start()
                let deadline = ContinuousClock.now + timeout
                while process.isRunning {
                    if ContinuousClock.now >= deadline {
                        await stopAndReap(process)
                        try Task.checkCancellation()
                        return "检查模型文件超时。请确认文件已下载到本机、磁盘已连接后重试。"
                    }
                    try await Task.sleep(for: .milliseconds(20))
                }
                try Task.checkCancellation()
                guard process.terminationStatus == 0 else {
                    return "无法检查模型文件，请重试。"
                }
                let response = try Data(contentsOf: directory.appendingPathComponent("response.json"))
                return try JSONDecoder().decode(String?.self, from: response)
            } onCancel: { process.stop() }
        } catch {
            await stopAndReap(process)
            throw error
        }
    }

    private func stopAndReap(_ process: StudioHelperProcess) async {
        process.stop()
        guard process.isRunning else { return }
        // Cleanup gets a brief independent wait even if the request task was cancelled.
        await Task.detached {
            let deadline = ContinuousClock.now + .seconds(1)
            while process.isRunning && ContinuousClock.now < deadline {
                try? await Task.sleep(for: .milliseconds(20))
            }
        }.value
    }
}
