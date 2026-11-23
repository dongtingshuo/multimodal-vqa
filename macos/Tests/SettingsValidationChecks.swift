import Darwin
import Foundation

@MainActor @main enum SettingsValidationChecks {
    static func main() async throws {
        if CommandLine.arguments.dropFirst().first == SettingsValidationWorker.argument {
            try holdSettingsWorkerIfRequested()
            SettingsValidationWorker.runIfRequested()
        }

        let temporary = URL(fileURLWithPath: CommandLine.arguments[1])
        let project = temporary.appendingPathComponent("检查路径 Project", isDirectory: true)
        let module = project.appendingPathComponent("vqa_project", isDirectory: true)
        try FileManager.default.createDirectory(at: module, withIntermediateDirectories: true)
        let worker = try String(contentsOf: temporary.appendingPathComponent("worker/vqa_project/desktop_worker.py"), encoding: .utf8)
        let operation = "    operation = question if action == 'predict' else action"
        precondition(worker.contains(operation))
        let recoveryControl = """

            if action == 'load' and (root / 'reject-model-load').exists():
                print(json.dumps({'id': request['id'], 'ok': False, 'error': 'CONTROLLED_MODEL_LOAD_FAILURE'}), flush=True)
                continue
            if action == 'load' and (root / 'hold-model-load').exists():
                (root / 'model-load-entered').write_text('PROTOCOL_TEST_ONLY')
                while (root / 'hold-model-load').exists(): time.sleep(0.01)
        """
        try worker.replacingOccurrences(of: operation, with: operation + "\n" + recoveryControl)
            .write(to: module.appendingPathComponent("desktop_worker.py"), atomically: true, encoding: .utf8)
        try Data().write(to: module.appendingPathComponent("__init__.py"))
        for name in ["unused.yaml", "unused.pt"] { try Data().write(to: project.appendingPathComponent(name)) }

        let original = AppSettings(project: project.path, python: CommandLine.arguments[2], config: "unused.yaml", checkpoint: "unused.pt")
        let defaults = UserDefaults.standard.data(forKey: "modelSettings")
        defer { UserDefaults.standard.set(defaults, forKey: "modelSettings") }
        let store = StudioStore(directory: temporary.appendingPathComponent("SettingsLibrary"))
        defer { store.shutdown() }
        store.settings = original
        await store.loadModel()
        precondition(store.ready && !store.busy, "The protocol-only model must load after checking Unicode/space paths")
        let pidFile = project.appendingPathComponent("worker.pid")
        let modelPID = try Int32(String(contentsOf: pidFile, encoding: .utf8))!
        var updated = original
        updated.topk = 3

        let gate = project.appendingPathComponent("hold-settings-validation")
        let marker = project.appendingPathComponent("settings-validation-entered.json")
        try Data().write(to: gate)
        let applying = Task { await store.apply(updated) }
        let stoppedHelper = try await waitForHelper(marker)
        precondition(store.ready && store.busy && store.checkingModelSettings && !exited(modelPID),
                     "The main actor must stay responsive with the previous model alive while a helper waits")
        await store.loadModel()
        let duplicate = await store.apply(updated)
        precondition(duplicate == nil && store.settings == original && UserDefaults.standard.data(forKey: "modelSettings") == defaults,
                     "Other model actions must not bypass a pending check")
        store.cancelModelTask()
        let stoppedIssue = await applying.value
        precondition(stoppedIssue == nil && store.ready && !store.busy && !store.checkingModelSettings && !exited(modelPID),
                     "Stopping settings checks must preserve the loaded model and leave the form usable")
        precondition(store.settings == original && UserDefaults.standard.data(forKey: "modelSettings") == defaults,
                     "Cancelling a settings check must leave saved settings unchanged")
        precondition(stoppedHelper.cleaned, "The cancelled helper and its private staging directory must be gone")

        try FileManager.default.removeItem(at: marker)
        let timed = Task.detached { try await SettingsValidationClient(timeout: .seconds(1)).message(for: original) }
        let timedHelper = try await waitForHelper(marker)
        let timedIssue = try await timed.value
        precondition(timedIssue?.contains("超时") == true && timedHelper.cleaned && !exited(modelPID),
                     "A timeout must explain the unavailable path, reap its helper and leave the model alive")

        try FileManager.default.removeItem(at: marker)
        let retry = Task { await store.apply(updated) }
        let releasedHelper = try await waitForHelper(marker)
        try FileManager.default.removeItem(at: gate)
        let retryIssue = await retry.value
        let currentPID = try Int32(String(contentsOf: pidFile, encoding: .utf8))!
        let savedSettings = UserDefaults.standard.data(forKey: "modelSettings")
        precondition(retryIssue == nil && store.settings == updated && store.ready && releasedHelper.cleaned,
                     "Retry must commit the current settings only after a successful check")
        precondition(currentPID == modelPID && !exited(modelPID) && savedSettings != defaults,
                     "A candidate-count-only update must preserve the same live model")

        let config = project.appendingPathComponent("unused.yaml")
        try FileManager.default.removeItem(at: config)
        await store.loadModel()
        precondition(store.errorMessage?.contains("配置") == true && store.ready && !store.busy && !exited(modelPID),
                     "A missing file on explicit reload must leave the existing model ready")
        try Data().write(to: config)

        await store.loadModel()
        precondition(store.ready && !store.modelError && store.errorEvent == nil && exited(modelPID),
                     "Successful reload must clear the resolved model validation error and reap the old worker")

        let rejection = project.appendingPathComponent("reject-model-load")
        try Data().write(to: rejection)
        let failedApply = await store.apply(updated)
        precondition(failedApply == "CONTROLLED_MODEL_LOAD_FAILURE" && !store.ready && store.modelError
                     && store.errorMessage == failedApply,
                     "Applying valid paths with a rejected model load must return the actual failure to Settings")
        let rejectedPID = try Int32(String(contentsOf: pidFile, encoding: .utf8))!
        try FileManager.default.removeItem(at: rejection)
        let recoveredApply = await store.apply(updated)
        precondition(recoveredApply == nil && store.ready && !store.modelError && store.errorEvent == nil && exited(rejectedPID),
                     "Successful apply must clear only its resolved model failure and reap the rejected worker")

        try FileManager.default.removeItem(at: config)
        await store.loadModel()
        let previousError = store.errorEvent!
        try Data().write(to: config)
        let modelGate = project.appendingPathComponent("hold-model-load")
        let modelEntered = project.appendingPathComponent("model-load-entered")
        try Data().write(to: modelGate)
        let recovering = Task { await store.loadModel() }
        let modelDeadline = ContinuousClock.now + .seconds(5)
        while !FileManager.default.fileExists(atPath: modelEntered.path) && ContinuousClock.now < modelDeadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(store.busy && FileManager.default.fileExists(atPath: modelEntered.path),
                     "The controlled model reload must be held while the main actor remains responsive")
        store.modelError = false
        store.errorMessage = previousError.message
        let newerError = store.errorEvent!
        try FileManager.default.removeItem(at: modelGate)
        await recovering.value
        precondition(store.ready && !store.modelError && store.errorEvent == newerError && newerError.id != previousError.id,
                     "Successful model recovery must preserve a newer unrelated event even when its text matches")
        var invalid = updated
        invalid.config = "missing.yaml"
        let invalidApply = await store.apply(invalid)
        precondition(invalidApply?.contains("配置") == true && store.errorEvent == newerError && store.ready,
                     "A form validation failure must preserve a separate newer error and the loaded model")
        precondition(store.dismissError(newerError) && store.errorEvent == nil,
                     "The preserved event must remain acknowledgeable after model recovery")
        let activePID = try Int32(String(contentsOf: pidFile, encoding: .utf8))!
        let beforeQuitSettings = UserDefaults.standard.data(forKey: "modelSettings")

        try FileManager.default.removeItem(at: marker)
        try Data().write(to: gate)
        let interrupted = Task { await store.apply(original) }
        let quitHelper = try await waitForHelper(marker)
        let terminated = await store.prepareToTerminate()
        _ = await interrupted.value
        precondition(terminated && quitHelper.cleaned && exited(activePID),
                     "Successful quit must reap the check and the Python model")
        precondition(store.settings == updated && UserDefaults.standard.data(forKey: "modelSettings") == beforeQuitSettings,
                     "A check cancelled by quit must never commit its settings")
        print("20 settings responsiveness/recovery checks passed (held native helper and isolated protocol model)")
    }

    private struct Helper: Codable {
        let pid: Int32
        let directory: URL
        var cleaned: Bool { SettingsValidationChecks.exited(pid) && !FileManager.default.fileExists(atPath: directory.path) }
    }

    private nonisolated static func exited(_ pid: Int32) -> Bool { Darwin.kill(pid, 0) != 0 && errno == ESRCH }

    private static func waitForHelper(_ marker: URL) async throws -> Helper {
        let deadline = ContinuousClock.now + .seconds(5)
        while ContinuousClock.now < deadline {
            if let data = try? Data(contentsOf: marker) { return try JSONDecoder().decode(Helper.self, from: data) }
            try await Task.sleep(for: .milliseconds(10))
        }
        preconditionFailure("The test helper must enter its controlled wait")
    }

    private static func holdSettingsWorkerIfRequested() throws {
        let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        let request = try Data(contentsOf: directory.appendingPathComponent("request.json"))
        let settings = try JSONDecoder().decode(AppSettings.self, from: request)
        let project = URL(fileURLWithPath: settings.project, isDirectory: true)
        let gate = project.appendingPathComponent("hold-settings-validation")
        guard FileManager.default.fileExists(atPath: gate.path) else { return }
        let helper = Helper(pid: getpid(), directory: directory)
        let marker = project.appendingPathComponent("settings-validation-entered.json")
        try JSONEncoder().encode(helper).write(to: marker, options: .atomic)
        while FileManager.default.fileExists(atPath: gate.path) { usleep(10_000) }
    }
}
