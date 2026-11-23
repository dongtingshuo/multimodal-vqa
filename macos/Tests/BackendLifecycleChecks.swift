import AppKit
import Darwin

@MainActor @main enum BackendLifecycleChecks {
    private static var checks = 0
    private static func check(_ value: @autoclosure () -> Bool, _ message: String) {
        precondition(value(), message); checks += 1
    }

    static func main() async throws {
        SettingsValidationWorker.runIfRequested()
        if ImageImportWorker.runIfRequested() { return }
        if CommandLine.arguments.dropFirst().first == "--quit-child" {
            let root = URL(fileURLWithPath: CommandLine.arguments[2])
            let template = URL(fileURLWithPath: CommandLine.arguments[3])
            let (store, _) = try await makeStore(root, template: template, python: CommandLine.arguments[4])
            let saved = await store.prepareToTerminate()
            check(saved, "The child App must complete its normal successful-quit path")
            await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
            return
        }
        let temporary = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = temporary.appendingPathComponent("backend-lifecycle")
        let template = temporary.appendingPathComponent("worker/vqa_project/desktop_worker.py")
        let python = CommandLine.arguments[2]
        try await idleChecks(root.appendingPathComponent("idle"), template: template, python: python)
        try await retiredChecks(root.appendingPathComponent("retired"), template: template, python: python)
        try await activeChecks(root.appendingPathComponent("active"), template: template, python: python)
        try await failedSaveAndReloadChecks(root.appendingPathComponent("save-reload"), template: template, python: python)
        try await cancelledReloadChecks(root.appendingPathComponent("cancel-reload"), template: template, python: python)
        try await parentExitChecks(root.appendingPathComponent("parent-exit"), template: template, python: python)
        try await sheetTerminationChecks()
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) backend lifecycle checks passed (owned protocol workers; no real model or user-library writes)")
    }

    private static func settings(_ project: URL, template: URL, python: String) throws -> AppSettings {
        let module = project.appendingPathComponent("vqa_project")
        try FileManager.default.createDirectory(at: module, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: template, to: module.appendingPathComponent("desktop_worker.py"))
        try Data().write(to: module.appendingPathComponent("__init__.py"))
        for file in ["unused.yaml", "idle-load.yaml", "unused.pt"] { try Data().write(to: project.appendingPathComponent(file)) }
        return AppSettings(project: project.path, python: python, config: "unused.yaml", checkpoint: "unused.pt")
    }

    private static func pid(_ settings: AppSettings) throws -> Int32 {
        try Int32(String(contentsOf: URL(fileURLWithPath: settings.project).appendingPathComponent("worker.pid"), encoding: .utf8))!
    }

    private static func exited(_ pid: Int32) -> Bool { Darwin.kill(pid, 0) != 0 && errno == ESRCH }

    private static func waitFor(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(condition(), "The owned protocol worker must reach the intended state")
    }

    private static func sheetTerminationChecks() async throws {
        let delegate = StudioAppDelegate()
        var continuation: CheckedContinuation<Bool, Never>?
        var attempts = 0
        var outcomes: [Bool] = []
        var completed = 0
        delegate.prepareToTerminate = {
            attempts += 1
            let result = await withCheckedContinuation { continuation = $0 }
            continuation = nil
            outcomes.append(result)
            return result
        }
        delegate.finishSheetTermination = { completed += 1 }
        delegate.requestSheetTermination()
        try await waitFor { continuation != nil }
        delegate.requestSheetTermination()
        check(attempts == 1 && completed == 0, "Repeated sheet quit must share the pending preparation")
        continuation?.resume(returning: false)
        try await waitFor { outcomes == [false] }
        check(completed == 0, "A failed save must keep the sheet open")
        delegate.requestSheetTermination()
        try await waitFor { continuation != nil && attempts == 2 }
        continuation?.resume(returning: true)
        try await waitFor { completed == 1 }
        check(outcomes == [false, true], "A failed sheet quit must allow a successful retry")
        delegate.requestSheetTermination()
        check(attempts == 2 && completed == 1, "A prepared sheet quit must complete exactly once")
    }

    private static func idleChecks(_ root: URL, template: URL, python: String) async throws {
        let settings = try settings(root, template: template, python: python)
        let client = BackendClient(logDirectory: root.appendingPathComponent("Logs"))
        _ = try await client.request(settings: settings, action: "load")
        let first = try pid(settings)
        await client.shutdownAndWait()
        check(exited(first), "Awaited shutdown must reap an idle worker before returning")
        await client.shutdownAndWait()
        let response = try await client.request(settings: settings, action: "idle-stubborn")
        let stubborn = try pid(settings)
        check(response.ok && stubborn != first, "An awaited shutdown remains reusable for explicit reload")
        let began = ContinuousClock.now
        client.shutdown()
        await client.shutdownAndWait()
        check(exited(stubborn) && began.duration(to: .now) < .seconds(3), "Awaited shutdown must also finish a previously requested stop of an idle SIGTERM-ignoring child")
    }

    private static func retiredChecks(_ root: URL, template: URL, python: String) async throws {
        for action in ["malformed-stubborn", "closed-output-stubborn"] {
            let settings = try settings(root.appendingPathComponent(action), template: template, python: python)
            let client = BackendClient(logDirectory: root.appendingPathComponent(action + "-logs"))
            do { _ = try await client.request(settings: settings, action: action); preconditionFailure("Broken protocol must fail") }
            catch let error as BackendError { check(error.needsReload, "A broken protocol must remain a connection error") }
            let child = try pid(settings)
            check(!exited(child), "The fixture must leave a retired worker alive until termination escalation")
            await client.shutdownAndWait()
            check(exited(child), "Connection cleanup must not lose ownership of a still-running retired child")
            let response = try await client.request(settings: settings, action: "load")
            let replacement = try pid(settings)
            check(response.ok && replacement != child, "Reload must work after fully draining a retired worker")
            await client.shutdownAndWait()
            check(exited(replacement), "Replacement workers must be reaped too")
        }
    }

    private static func activeChecks(_ root: URL, template: URL, python: String) async throws {
        let settings = try settings(root, template: template, python: python)
        let client = BackendClient(logDirectory: root.appendingPathComponent("Logs"))
        let request = Task { try await client.request(settings: settings, action: "stubborn") }
        let marker = root.appendingPathComponent("stubborn.pid")
        try await waitFor { FileManager.default.fileExists(atPath: marker.path) }
        let child = try pid(settings)
        let drain = Task { await client.shutdownAndWait() }
        drain.cancel()
        await drain.value
        check(exited(child), "Cancelling the caller must not skip mandatory owned-process cleanup")
        do { _ = try await request.value; preconditionFailure("Shutdown must interrupt active inference") }
        catch is CancellationError { check(true, "Active requests must complete as cancellation") }
        await client.shutdownAndWait()
    }

    private static func makeStore(_ root: URL, template: URL, python: String) async throws -> (StudioStore, AppSettings) {
        var settings = try settings(root.appendingPathComponent("Worker"), template: template, python: python)
        settings.config = "idle-load.yaml"
        let library = root.appendingPathComponent("Library")
        let store = StudioStore(directory: library)
        store.settings = settings
        await store.loadModel()
        check(store.ready && !store.busy, "The isolated store must be ready before the lifecycle action")
        return (store, settings)
    }

    private static func failedSaveAndReloadChecks(_ root: URL, template: URL, python: String) async throws {
        let (store, settings) = try await makeStore(root, template: template, python: python)
        let first = try pid(settings)
        let library = root.appendingPathComponent("Library")
        let snapshot = library.appendingPathComponent("library.json")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let failed = await store.prepareToTerminate()
        check(!failed && !store.quitting && store.ready && !exited(first), "A failed save must leave the existing loaded worker running and the App usable")
        try FileManager.default.removeItem(at: snapshot)
        store.retrySave()
        check(!store.hasUnsavedChanges, "Repairing storage must allow retry without touching the worker")
        store.settings.config = "unused.yaml"
        let reload = Task { await store.loadModel() }
        try await waitFor { store.busy && !store.ready }
        await store.loadModel()
        check(!exited(first) && store.busy, "A duplicate reload must not bypass the in-progress drain or launch a replacement early")
        await reload.value
        let replacement = try pid(settings)
        check(exited(first) && store.ready && replacement != first, "Explicit reload must finish the old worker before the new model becomes ready")
        let request = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("Worker/last-request.json"))) as! [String: Any]
        check(request["action"] as? String == "load", "A duplicate reload must not enqueue a second model request")
        let saved = await store.prepareToTerminate()
        check(saved && exited(replacement), "Successful quit after recovery must reap the replacement model")
    }

    private static func cancelledReloadChecks(_ root: URL, template: URL, python: String) async throws {
        let (store, settings) = try await makeStore(root, template: template, python: python)
        let child = try pid(settings)
        let reloading = Task { await store.loadModel() }
        try await waitFor { store.busy && !store.ready }
        store.cancelModelTask(); await reloading.value
        let after = try pid(settings)
        check(exited(child) && child == after && !store.ready && !store.busy && store.status == "模型已停止", "Stopping while reload drains a worker must not launch a new model afterward")
        store.settings.config = "unused.yaml"
        await store.loadModel()
        check(store.ready, "A deliberately stopped reload must remain retryable")
        let replacement = try pid(settings)
        let saved = await store.prepareToTerminate()
        check(saved && exited(replacement), "Retry must retain complete termination ownership")
    }

    private static func parentExitChecks(_ root: URL, template: URL, python: String) async throws {
        let parent = Process()
        parent.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        parent.arguments = ["--quit-child", root.path, template.path, python]
        parent.standardOutput = FileHandle.nullDevice
        parent.standardError = FileHandle.standardError
        try parent.run()
        let deadline = ContinuousClock.now.advanced(by: .seconds(10))
        while parent.isRunning && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        let completed = !parent.isRunning
        if !completed { _ = Darwin.kill(parent.processIdentifier, SIGKILL) }
        let marker = root.appendingPathComponent("Worker/worker.pid")
        let child = try Int32(String(contentsOf: marker, encoding: .utf8))!
        let reaped = exited(child)
        if !reaped { _ = Darwin.kill(child, SIGKILL) }
        check(completed && parent.terminationStatus == 0, "A real short-lived parent must finish quit without hanging")
        check(reaped, "After the App process exits, its stubborn model worker must already be gone")
        let restored = try LibraryStorage(directory: root.appendingPathComponent("Library")).load()
        check(restored.sessions.isEmpty, "Process-exit verification must preserve the independent library without invented records")
    }
}
