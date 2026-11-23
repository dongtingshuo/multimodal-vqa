import AppKit
import Darwin
import Foundation

@MainActor @main
enum DesktopChecks {
    private static var checks = 0

    static func main() async throws {
        if ImageImportWorker.runIfRequested() { return }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        try storageChecks(root.appendingPathComponent("storage"))
        try await storeChecks(root.appendingPathComponent("store"))
        try await importChecks(root.appendingPathComponent("imports"))
        queryAndExportChecks()
        try settingsChecks(root.appendingPathComponent("settings"))
        try await backendChecks(root.appendingPathComponent("worker"), python: CommandLine.arguments[2])
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) desktop regression checks passed (isolated temporary data; protocol test worker only)")
    }

    private static func check(_ value: @autoclosure () -> Bool, _ message: String) {
        precondition(value(), message)
        checks += 1
    }

    private static func sample(_ name: String = "Birds") -> ImageSession {
        ImageSession(title: name, imageName: UUID().uuidString + ".png", questions: [
            QuestionRecord(question: "What is visible?", answers: [Prediction(answer: "bird|branch", score: 0.75)], elapsed: 0.2, model: "test-fixture", checkpoint: "not-a-real-model", device: "cpu")
        ])
    }

    private static func storageChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let session = sample()
        let legacy = try JSONEncoder().encode([session])
        try legacy.write(to: root.appendingPathComponent("history.json"))
        let storage = LibraryStorage(directory: root)
        var state = try storage.load()
        check(state.sessions == [session], "Legacy history must migrate without losing any record metadata")
        state.drafts = [session.id.uuidString: "A draft that must survive restart"]
        state.favoriteIDs = [session.id]
        state.sortOrder = .name
        try storage.save(state)
        let restored = try storage.load()
        check(restored == state, "The whole workspace must round-trip atomically")
        let legacyAfterSave = try Data(contentsOf: root.appendingPathComponent("history.json"))
        check(legacyAfterSave == legacy, "The legacy history must remain untouched")
        try Data("{broken library".utf8).write(to: root.appendingPathComponent("library.json"))
        do { _ = try storage.load(); preconditionFailure("A corrupt new library must not silently fall back to stale history") }
        catch { checks += 1 }
    }

    private static func storeChecks(_ root: URL) async throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let original = sample()
        var initial = LibraryState(sessions: [original], selectedID: original.id)
        try LibraryStorage(directory: root).save(initial)
        let store = StudioStore(directory: root)
        store.question = "Keep this unsubmitted question"
        store.showInspector = false
        store.sortOrder = .name
        try await Task.sleep(for: .milliseconds(500))
        let autosaved = try LibraryStorage(directory: root).load()
        check(autosaved.drafts[original.id.uuidString] == "Keep this unsubmitted question", "Drafts must autosave without needing quit")
        store.toggleFavorite(original.id)
        store.rename(original.id, to: "  Renamed birds  ")
        check(store.selected?.title == "Renamed birds", "Rename trims whitespace and preserves identity")
        check(store.selected?.questions == original.questions, "Rename must preserve answer history")
        store.remove(original.id)
        check(store.sessions.isEmpty && store.archived.first?.id == original.id, "Removing an image must archive it")
        store.shutdown()
        let relaunched = StudioStore(directory: root)
        check(relaunched.archived.count == 1, "Recently deleted items must survive restart")
        relaunched.restoreArchived(original.id)
        check(relaunched.selected?.questions == original.questions, "Restore must preserve all questions")
        check(relaunched.question == "Keep this unsubmitted question", "Restore must recover the image's draft")
        check(relaunched.favoriteIDs.contains(original.id), "Restore must keep the favorite flag")
        check(!relaunched.showInspector && relaunched.sortOrder == .name, "Workspace preferences must survive restart")
        relaunched.pendingQuestion = PendingQuestion(sessionID: original.id, question: "Pending question")
        relaunched.remove(original.id)
        check(relaunched.sessions.count == 1, "An image currently being analyzed must not be archived")
        relaunched.question = ""
        relaunched.shutdown()
        let interrupted = try LibraryStorage(directory: root).load()
        check(interrupted.drafts[original.id.uuidString] == "Pending question", "Quit during inference must preserve the unfinished question")
        relaunched.pendingQuestion = nil

        let corruptRoot = root.appendingPathComponent("corrupt")
        try FileManager.default.createDirectory(at: corruptRoot, withIntermediateDirectories: true)
        let bytes = Data("not valid JSON".utf8)
        try bytes.write(to: corruptRoot.appendingPathComponent("library.json"))
        let protected = StudioStore(directory: corruptRoot)
        protected.question = "Do not overwrite the corrupt source"
        protected.shutdown()
        check(!protected.canModifyLibrary, "A corrupt library must become read-only")
        let protectedBytes = try Data(contentsOf: corruptRoot.appendingPathComponent("library.json"))
        check(protectedBytes == bytes, "A corrupt original must never be overwritten")

        let unwritable = root.appendingPathComponent("not-a-directory")
        try Data("file".utf8).write(to: unwritable)
        let failing = StudioStore(directory: unwritable)
        initial.sessions = [original]
        failing.sessions = initial.sessions
        failing.selectedID = original.id
        failing.rename(original.id, to: "Must roll back")
        check(failing.sessions.first?.title == original.title, "A failed rename save must roll back the in-memory change")
        failing.shutdown()
    }

    private static func importChecks(_ root: URL) async throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 3, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                      colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 255, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        let png = bitmap.representation(using: .png, properties: [:])!
        let first = root.appendingPathComponent("first.png"), second = root.appendingPathComponent("second.png")
        let invalid = root.appendingPathComponent("invalid.png")
        try png.write(to: first); try png.write(to: second); try Data("not an image".utf8).write(to: invalid)
        let store = StudioStore(directory: root.appendingPathComponent("library"))
        store.search = "Hidden filter"; store.favoritesOnly = true
        store.importImages([first, invalid])
        store.importImages([second])
        let limit = ContinuousClock.now.advanced(by: .seconds(5))
        while store.importing && ContinuousClock.now < limit { try await Task.sleep(for: .milliseconds(10)) }
        check(!store.importing && store.sessions.count == 2, "Queued batch imports must continue after one invalid image")
        check(store.selected?.title == "second", "The last imported image must be selected")
        check(store.search.isEmpty && !store.favoritesOnly, "Import must reveal the new image by clearing filters")
        check(store.errorMessage?.contains("1 张") == true, "Partial failures must be reported without losing successful imports")
        let originalBytes = try Data(contentsOf: first)
        check(originalBytes == png, "Original image files must remain untouched")
        let ownedURL = store.imageURL!
        store.remove(store.selectedID!)
        check(FileManager.default.fileExists(atPath: ownedURL.path), "Archiving must keep the owned image file recoverable")
        store.shutdown()
    }

    private static func queryAndExportChecks() {
        var first = sample("Zebra"), second = sample("Apple")
        first.date = Date(timeIntervalSince1970: 100); second.date = Date(timeIntervalSince1970: 200)
        first.questions[0].date = Date(timeIntervalSince1970: 300)
        second.questions[0].date = Date(timeIntervalSince1970: 250)
        check(LibraryQuery.results([first, second], query: "  BRANCH  ", favoritesOnly: false, favorites: [], sort: .name).map(\.title) == ["Apple", "Zebra"], "Search must include answer text and trim whitespace")
        check(LibraryQuery.results([first, second], query: "", favoritesOnly: true, favorites: [first.id], sort: .name).map(\.id) == [first.id], "Favorites filtering must preserve image identity")
        check(LibraryQuery.results([first, second], query: "", favoritesOnly: false, favorites: [], sort: .recentQuestions).first?.id == first.id, "Recent-question sorting must use the actual question date")
        check(LibraryQuery.results([first, second], query: "", favoritesOnly: false, favorites: [], sort: .imported).first?.id == second.id, "Import sorting must use the original import date")
        let report = ConversationExporter.markdown(first)
        check(report.contains("bird\\|branch") && report.contains("75.0%"), "Readable exports must escape table pipes and format scores")
        check(report.contains("分数不代表正确率"), "Readable exports must preserve the model-score limitation")
        check(ImageZoom.quadruple.larger == .quadruple && ImageZoom.quarter.smaller == .quarter, "Zoom controls must stop at their limits")
        check(ImageZoom.fit.larger == .actual && ImageZoom.fit.smaller == .half, "Fit mode must have predictable zoom steps")
    }

    private static func settingsChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("vqa_project"), withIntermediateDirectories: true)
        for name in ["vqa_project/desktop_worker.py", "python", "config.yaml", "weights.pt"] {
            try Data("filesystem validation fixture only".utf8).write(to: root.appendingPathComponent(name))
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: root.appendingPathComponent("python").path)
        var settings = AppSettings(project: root.path, python: root.appendingPathComponent("python").path, config: "config.yaml", checkpoint: "weights.pt")
        check(settings.validationMessage == nil, "Valid local paths must pass filesystem validation")
        settings.config = ""
        check(settings.validationMessage != nil, "An empty config must not resolve to a valid project directory")
        settings.config = "."
        check(settings.validationMessage != nil, "A directory must not be accepted as a config file")
        settings.config = "config.yaml"; settings.python = root.path
        check(settings.validationMessage != nil, "A searchable directory must not be accepted as a Python executable")
        settings.python = root.appendingPathComponent("python").path; settings.checkpoint = "missing.pt"
        check(settings.validationMessage != nil, "Missing weights must be reported before replacing active settings")
        let pipe = root.appendingPathComponent("input.pipe")
        precondition(mkfifo(pipe.path, 0o755) == 0)
        let pythonLink = root.appendingPathComponent("python-link")
        try FileManager.default.createSymbolicLink(at: pythonLink, withDestinationURL: root.appendingPathComponent("python"))
        let configLink = root.appendingPathComponent("config-link.yaml")
        try FileManager.default.createSymbolicLink(at: configLink, withDestinationURL: root.appendingPathComponent("config.yaml"))
        settings.checkpoint = "weights.pt"; settings.config = "config-link.yaml"; settings.python = pythonLink.path
        check(settings.validationMessage == nil, "Links to regular readable files and executables must still work")
        settings.config = "input.pipe"
        check(settings.validationMessage?.contains("配置") == true, "A readable pipe must not be accepted as a YAML file")
        let pipeLink = root.appendingPathComponent("pipe-link.pt")
        try FileManager.default.createSymbolicLink(at: pipeLink, withDestinationURL: pipe)
        settings.config = "config.yaml"; settings.checkpoint = "pipe-link.pt"
        check(settings.validationMessage?.contains("权重") == true, "A symlink to a pipe must not be accepted as a checkpoint")
        settings.checkpoint = "weights.pt"; settings.python = pipe.path
        check(settings.validationMessage?.contains("Python") == true, "An executable-bit pipe must not be accepted as Python")
    }

    private static func backendChecks(_ project: URL, python: String) async throws {
        let client = BackendClient(logDirectory: project.appendingPathComponent("logs"))
        defer { client.shutdown() }
        let settings = AppSettings(project: project.path, python: python, config: "unused.yaml", checkpoint: "unused.pt")
        let waiting = Task { try await client.request(settings: settings, action: "wait") }
        let marker = project.appendingPathComponent("waiting.pid")
        let limit = ContinuousClock.now.advanced(by: .seconds(5))
        while !FileManager.default.fileExists(atPath: marker.path) && ContinuousClock.now < limit { try await Task.sleep(for: .milliseconds(10)) }
        check(FileManager.default.fileExists(atPath: marker.path), "The protocol test worker must actually start before cancellation")
        let oldPID = Int32(try String(contentsOf: marker, encoding: .utf8))!
        let start = ContinuousClock.now
        client.cancel()
        do { _ = try await waiting.value; preconditionFailure("A cancelled worker request must not return a result") }
        catch is CancellationError { checks += 1 }
        check(start.duration(to: .now) < .seconds(3), "Cancellation must interrupt a blocked read promptly")
        let restarted = try await client.request(settings: settings, action: "load")
        check(restarted.ok && restarted.model == "PROTOCOL_TEST_ONLY", "A request after cancellation must start a usable new worker")
        let exitLimit = ContinuousClock.now.advanced(by: .seconds(2))
        while kill(oldPID, 0) == 0 && ContinuousClock.now < exitLimit { try await Task.sleep(for: .milliseconds(10)) }
        check(kill(oldPID, 0) != 0, "The cancelled worker must not remain orphaned")
        do { _ = try await client.request(settings: settings, action: "mismatch"); preconditionFailure("Mismatched response IDs must be rejected") }
        catch { checks += 1 }
        let recovered = try await client.request(settings: settings, action: "load")
        check(recovered.ok, "A rejected response must not corrupt the next request")
        client.shutdown()
        let afterShutdown = try await client.request(settings: settings, action: "load")
        check(afterShutdown.ok, "Shutdown followed by model reload must serialize cleanup correctly")
    }
}
