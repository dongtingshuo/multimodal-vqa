import AppKit
import Darwin
import Foundation

@MainActor @main
enum SnapshotWorkflowChecks {
    private static var checks = 0
    private static func check(_ value: @autoclosure () -> Bool, _ message: String) {
        precondition(value(), message); checks += 1
    }

    static func main() async throws {
        SettingsValidationWorker.runIfRequested()
        if CommandLine.arguments.dropFirst().first == ImageImportWorker.argument {
            try blockNativeInputIfRequested()
            if ImageImportWorker.runIfRequested() { return }
        }
        let temporary = URL(fileURLWithPath: CommandLine.arguments[1])
        let root = temporary.appendingPathComponent("snapshot-workflows")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let settings = AppSettings(project: temporary.appendingPathComponent("worker").path,
                                   python: CommandLine.arguments[2], config: "unused.yaml", checkpoint: "unused.pt")
        try await identityChecks(root.appendingPathComponent("identity"))
        try await preparationChecks(root.appendingPathComponent("prepare"), settings: settings)
        try await timeoutPreparationChecks(root.appendingPathComponent("prepare-timeout"), settings: settings)
        try await quitPreparationChecks(root.appendingPathComponent("quit-prepare"), settings: settings)
        try await quitInferenceChecks(root.appendingPathComponent("quit-inference"), settings: settings)
        try await quitImportChecks(root.appendingPathComponent("quit-import"))
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) snapshot/quit regression checks passed (isolated data; real native FIFO blocking; protocol-only model)")
    }

    private static func png(_ url: URL, width: Int = 4) throws -> Data {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: 3, bitsPerSample: 8,
                                     samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                     bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 180, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        let data = bitmap.representation(using: .png, properties: [:])!
        try data.write(to: url, options: .atomic)
        return data
    }

    private static func gate(_ source: URL, mode: String) throws {
        try Data().write(to: source.appendingPathExtension(mode + "-gate"))
        precondition(mkfifo(source.appendingPathExtension(mode + "-fifo").path, 0o600) == 0)
    }

    private static func blockNativeInputIfRequested() throws {
        let job = URL(fileURLWithPath: CommandLine.arguments[2])
        let payload = try JSONSerialization.jsonObject(with: Data(contentsOf: job.appendingPathComponent("request.json"))) as! [String: String]
        let source = URL(fileURLWithPath: payload["path"]!)
        let mode = payload["mode"] == "snapshot" ? "snapshot" : payload["mode"] == nil ? "import" : "read"
        guard FileManager.default.fileExists(atPath: source.appendingPathExtension(mode + "-gate").path) else { return }
        signal(SIGTERM, SIG_IGN)
        // Simulate a partial native copy inside the private job only; never touch the source.
        let partial = job.appendingPathComponent("Partial")
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: false)
        try Data("unfinished fixture output".utf8).write(to: partial.appendingPathComponent("input"))
        try Data(job.path.utf8).write(to: source.appendingPathExtension(mode + "-job"))
        try Data(String(getpid()).utf8).write(to: source.appendingPathExtension(mode + "-pid"))
        let fd = Darwin.open(source.appendingPathExtension(mode + "-fifo").path, O_RDONLY)
        if fd >= 0 { var byte: UInt8 = 0; _ = Darwin.read(fd, &byte, 1); _ = Darwin.close(fd) }
    }

    private static func waitFor(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while !condition() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(condition(), "The isolated operation must reach its intended state before continuing")
    }

    private static func blockedPID(_ source: URL, mode: String = "snapshot") async throws -> Int32 {
        let marker = source.appendingPathExtension(mode + "-pid")
        try await waitFor { FileManager.default.fileExists(atPath: marker.path) }
        return try Int32(String(contentsOf: marker, encoding: .utf8))!
    }

    private static func reaped(_ pid: Int32) async throws {
        try await waitFor { Darwin.kill(pid, 0) != 0 && errno == ESRCH }
    }

    private static func assertJobRemoved(_ source: URL, mode: String = "snapshot") throws {
        let path = try String(contentsOf: source.appendingPathExtension(mode + "-job"), encoding: .utf8)
        check(!FileManager.default.fileExists(atPath: path), "A cancelled/failed operation must remove its partial job directory")
    }

    private static func identityChecks(_ root: URL) async throws {
        let source = root.appendingPathComponent("source.png")
        let original = try png(source)
        let client = ImageImportClient(temporaryDirectory: root.appendingPathComponent("Staging"), timeout: .milliseconds(800))
        let stamp = try ImageFileStamp.read(source)
        let input = try await client.snapshot(at: source, stamp: stamp)
        let copied = try Data(contentsOf: input.url)
        check(input.url != source && copied == original, "Prepared input must be an independent byte-identical file, without re-encoding")
        _ = try png(source, width: 8)
        let retained = try ImageCodec.decode(input.url)
        check(retained.width == 4, "Replacing the source must not alter a prepared model input")
        do { _ = try await client.snapshot(at: source, stamp: stamp); preconditionFailure("Stale preview must fail") }
        catch ImageSnapshotError.changed { check(true, "A changed source must retain the typed snapshot error") }
        input.remove()
        check(!FileManager.default.fileExists(atPath: input.url.path), "Releasing a prepared input must remove its retained job output")
        let link = root.appendingPathComponent("linked.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        let linked = try await client.snapshot(at: link, stamp: ImageFileStamp.read(link))
        let kind = try FileManager.default.attributesOfItem(atPath: linked.url.path)[.type] as? FileAttributeType
        check(kind == .typeRegular, "A symbolic link must become a regular private input file")
        linked.remove()
        try gate(source, mode: "snapshot")
        let currentStamp = try ImageFileStamp.read(source)
        let timed = Task { try await client.snapshot(at: source, stamp: currentStamp) }
        let pid = try await blockedPID(source)
        do { _ = try await timed.value; preconditionFailure("Blocked snapshot must time out") }
        catch ImageSnapshotError.unavailable(let message) { check(message.contains("超时"), "Snapshot timeout must be an input failure, not a model connection failure") }
        try await reaped(pid); try assertJobRemoved(source)
        let files = try FileManager.default.contentsOfDirectory(atPath: client.temporaryDirectory.path)
        check(files.isEmpty, "Successful, rejected and timed-out snapshots must release their job outputs")
    }

    private static func makeStore(_ root: URL, settings: AppSettings, preparationClient: ImageImportClient? = nil) async throws -> (StudioStore, URL) {
        let session = ImageSession(title: "Snapshot fixture", imageName: "source.png")
        let source = root.appendingPathComponent("Images/source.png")
        _ = try png(source)
        try LibraryStorage(directory: root).save(LibraryState(sessions: [session], selectedID: session.id))
        let store: StudioStore
        if let preparationClient {
            store = StudioStore(directory: root, imagePreparer: { try await preparationClient.snapshot(at: $0, stamp: $1) })
        } else { store = StudioStore(directory: root) }
        store.settings = settings
        await store.loadModel(); await store.preview.waitUntilLoaded()
        check(store.ready && store.preview.image != nil, "The fixture must have a ready protocol model and a real decoded preview")
        return (store, source)
    }

    private static func modelPID(_ settings: AppSettings) throws -> Int {
        try Int(String(contentsOf: URL(fileURLWithPath: settings.project).appendingPathComponent("worker.pid"), encoding: .utf8))!
    }

    private static func preparationChecks(_ root: URL, settings: AppSettings) async throws {
        let (store, source) = try await makeStore(root, settings: settings)
        defer { store.shutdown() }
        let pidBefore = try modelPID(settings)
        try gate(source, mode: "snapshot")
        store.question = "Submitted question"
        let ask = Task { await store.ask() }
        let pid = try await blockedPID(source)
        check(store.preparingImage && store.busy, "The UI must distinguish preparing pixels from running inference")
        store.question = "A newer draft"
        let start = ContinuousClock.now; store.cancelModelTask(); await ask.value
        check(start.duration(to: .now) < .seconds(1) && !store.busy && !store.cancelling && !store.preparingImage, "Stop must end blocked preparation without releasing the native input")
        check(store.ready && store.question == "A newer draft" && store.recoverableQuestions.contains(where: { $0.question == "Submitted question" }), "Preparation cancellation must keep the model ready and preserve both submitted text and newer typing")
        check(store.selected?.questions.isEmpty == true && store.errorMessage == nil, "Stopping preparation must not create answers or a model error")
        try await reaped(pid); try assertJobRemoved(source)
        try FileManager.default.removeItem(at: source.appendingPathExtension("snapshot-gate"))
        store.recoverQuestion(store.recoverableQuestions[0].id)
        await store.ask()
        let pidAfter = try modelPID(settings)
        check(store.selected?.questions.count == 1 && store.ready && pidBefore == pidAfter, "Retry must answer through the same model process without reloading")
        check(store.recoverableQuestions.contains(where: { $0.question == "A newer draft" }), "Retrying the submitted question must retain the newer draft")
    }

    private static func timeoutPreparationChecks(_ root: URL, settings: AppSettings) async throws {
        let client = ImageImportClient(temporaryDirectory: root.appendingPathComponent("Staging"), timeout: .milliseconds(800))
        let (store, source) = try await makeStore(root, settings: settings, preparationClient: client)
        defer { store.shutdown() }
        let before = try modelPID(settings)
        try gate(source, mode: "snapshot")
        store.question = "Retry after input timeout"
        let ask = Task { await store.ask() }; let pid = try await blockedPID(source)
        await ask.value; await store.preview.waitUntilLoaded()
        check(store.ready && !store.busy && store.question == "Retry after input timeout" && store.errorMessage?.contains("超时") == true,
              "A snapshot timeout must retain model readiness, restore the question and explain the input failure")
        try await reaped(pid); try assertJobRemoved(source)
        try FileManager.default.removeItem(at: source.appendingPathExtension("snapshot-gate"))
        store.errorMessage = nil; await store.ask()
        let after = try modelPID(settings)
        check(store.selected?.questions.count == 1 && before == after, "Retry after input timeout must use the original model process")
    }

    private static func quitPreparationChecks(_ root: URL, settings: AppSettings) async throws {
        let (store, source) = try await makeStore(root, settings: settings)
        try gate(source, mode: "snapshot")
        store.question = "Keep this after quitting preparation"
        let ask = Task { await store.ask() }; let pid = try await blockedPID(source)
        let start = ContinuousClock.now; let quit = await store.prepareToTerminate(); await ask.value
        check(quit && start.duration(to: .now) < .seconds(1), "Quit must cancel blocked snapshot preparation instead of waiting for its deadline")
        try await reaped(pid); try assertJobRemoved(source)
        let reopened = StudioStore(directory: root)
        check(reopened.question == "Keep this after quitting preparation" && reopened.selected?.questions.isEmpty == true, "Restart must recover a question interrupted before model dispatch")
        reopened.shutdown()
    }

    private static func quitInferenceChecks(_ root: URL, settings: AppSettings) async throws {
        let (store, _) = try await makeStore(root, settings: settings)
        let marker = URL(fileURLWithPath: settings.project).appendingPathComponent("stubborn.pid")
        try? FileManager.default.removeItem(at: marker)
        store.question = "stubborn"
        let ask = Task { await store.ask() }
        try await waitFor { FileManager.default.fileExists(atPath: marker.path) }
        let request = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: settings.project).appendingPathComponent("last-request.json"))) as! [String: Any]
        let privateInput = request["image"] as! String
        check(FileManager.default.fileExists(atPath: privateInput), "The model must retain its private input until completion or termination")
        let quit = await store.prepareToTerminate()
        check(quit && !FileManager.default.fileExists(atPath: privateInput), "Successful quit must remove active inference input before the asynchronous model callback")
        await ask.value
        check(store.question == "stubborn" && store.selected?.questions.isEmpty == true, "A quit during inference must retain the question without inventing an answer")
    }

    private static func quitImportChecks(_ root: URL) async throws {
        let source = root.appendingPathComponent("accepted.png"), blocked = root.appendingPathComponent("blocked.png")
        let original = try png(source); _ = try png(blocked); try gate(blocked, mode: "import")
        let library = root.appendingPathComponent("Library")
        let store = StudioStore(directory: library)
        store.stopImportsForQuit()
        store.importImages([source, blocked, source]); let pid = try await blockedPID(blocked, mode: "import")
        check(store.sessions.count == 1, "The first import must already be committed before the quit test")
        store.question = "Keep the imported image draft"
        let snapshot = library.appendingPathComponent("library.json")
        let saved = try Data(contentsOf: snapshot)
        try FileManager.default.removeItem(at: snapshot); try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let quit = Task { await store.prepareToTerminate() }
        try await waitFor { store.quitting }
        store.cancelImport()
        check(!store.stoppingImport, "Ordinary Stop must not silently change the default quit-drain promise")
        let start = ContinuousClock.now; store.stopImportsForQuit(); store.stopImportsForQuit()
        let failed = await quit.value
        check(!failed && !store.quitting && start.duration(to: .now) < .seconds(1) && store.hasUnsavedChanges, "An explicit stop-and-quit must cancel promptly, but a failed save must still keep the App open")
        check(store.sessions.count == 1 && store.question == "Keep the imported image draft", "Stopping the quit queue must retain completed images and drafts")
        try await reaped(pid); try assertJobRemoved(blocked, mode: "import")
        try FileManager.default.removeItem(at: snapshot); try saved.write(to: snapshot)
        store.retrySave(); store.importImages([source]); try await waitFor { !store.importing }
        check(store.sessions.count == 2 && !store.hasUnsavedChanges, "The App must remain usable after a stopped import and failed quit are repaired")
        let finalQuit = await store.prepareToTerminate()
        check(finalQuit, "Normal quit must still work after repairing the failed save")
        let originalAfter = try Data(contentsOf: source)
        check(originalAfter == original, "Stop-and-quit must never delete or modify original source files")
    }
}
