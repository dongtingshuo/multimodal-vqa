import AppKit
import Darwin
import Foundation

@MainActor @main
enum ReliabilityChecks {
    private static var checks = 0
    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message)
        checks += 1
    }

    static func main() async throws {
        SettingsValidationWorker.runIfRequested()
        if ImageImportWorker.runIfRequested() { return }
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let project = root.appendingPathComponent("worker")
        let settings = AppSettings(project: project.path, python: CommandLine.arguments[2], config: "unused.yaml", checkpoint: "unused.pt")
        try await iterationRecoveryChecks(root.appendingPathComponent("iteration-recovery"))
        try integrityChecks(root.appendingPathComponent("integrity"))
        try authorityChecks(root.appendingPathComponent("authority"))
        try conflictChecks(root.appendingPathComponent("conflicts"))
        fileOpeningChecks(root)
        try settingsAndExportChecks(root.appendingPathComponent("settings"))
        try await persistenceChecks(root.appendingPathComponent("persistence"), settings: settings)
        try unfinishedQuestionChecks(root.appendingPathComponent("unfinished"))
        try await settingsApplicationChecks(root.appendingPathComponent("settings-apply"), settings: settings)
        try await bootstrapChecks(root.appendingPathComponent("带 空格的 worker"), settings: settings)
        try await startupLifetimeChecks(root.appendingPathComponent("startup-lifetime"), settings: settings)
        try await transportChecks(project, settings: settings)
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) reliability regression checks passed (isolated data; protocol test worker only)")
    }

    private static func rejects(_ operation: () throws -> Void, _ message: String) {
        do { try operation(); preconditionFailure(message) }
        catch { checks += 1 }
    }

    private static func iterationRecoveryChecks(_ root: URL) async throws {
        var failures: [String] = []
        func verify(_ value: Bool, _ message: String) {
            checks += 1
            if !value { failures.append(message); print("REGRESSION: \(message)") }
        }
        let first = ImageSession(title: "First", imageName: "first.png")
        let second = ImageSession(title: "Second", imageName: "second.png")
        let initial = LibraryState(sessions: [first], archived: [ArchivedImage(session: second)],
                                   favoriteIDs: [first.id], selectedID: first.id, favoritesOnly: true)
        let rollbackRoot = root.appendingPathComponent("rollback")
        try LibraryStorage(directory: rollbackRoot).save(initial)
        let rollback = StudioStore(directory: rollbackRoot)
        rollback.search = "First"; rollback.imageZoom = .double
        var external = initial; external.sessions[0].title = "External edit"
        try JSONEncoder().encode(external).write(to: rollbackRoot.appendingPathComponent("library.json"), options: .atomic)
        rollback.restoreArchived(second.id)
        verify(rollback.search == "First" && rollback.favoritesOnly, "Failed restore must preserve the search and favorites filter")
        verify(rollback.selectedID == first.id && rollback.imageZoom == .double && rollback.archived.count == 1, "Failed restore must preserve selection, zoom and archive contents")
        rollback.imageZoom = .quadruple
        rollback.remove(first.id)
        verify(rollback.selectedID == first.id && rollback.imageZoom == .quadruple, "Failed archive must preserve the current image zoom")
        rollback.shutdown()

        let recoveredRoot = root.appendingPathComponent("recovered-import")
        try LibraryStorage(directory: recoveredRoot).save(initial)
        let recovered = StudioStore(directory: recoveredRoot)
        let snapshot = recoveredRoot.appendingPathComponent("library.json")
        let original = try Data(contentsOf: snapshot)
        try FileManager.default.removeItem(at: snapshot)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        recovered.question = "Keep this unsaved draft"; recovered.retrySave()
        check(recovered.hasUnsavedChanges, "The fixture must enter a real failed-save state")
        try FileManager.default.removeItem(at: snapshot); try original.write(to: snapshot)
        let source = root.appendingPathComponent("source.png")
        try png().write(to: source)
        recovered.importImages([source])
        try await waitForImport(recovered)
        let saved = try LibraryStorage(directory: recoveredRoot).load()
        verify(!recovered.hasUnsavedChanges && recovered.errorMessage == nil && saved.sessions.count == 2, "Successful import after disk recovery must clear the stale unsaved warning and its obsolete alert")
        verify(saved.drafts[first.id.uuidString] == "Keep this unsaved draft", "Import recovery must durably retain the earlier unsaved draft")
        recovered.errorMessage = "Unrelated model error"; recovered.modelError = true
        recovered.importImages([source])
        try await waitForImport(recovered)
        verify(recovered.errorMessage == "Unrelated model error" && recovered.modelError,
               "A successful library save must not dismiss an unrelated model error")
        recovered.shutdown()

        let conflictRoot = root.appendingPathComponent("batch-conflict")
        try LibraryStorage(directory: conflictRoot).save(initial)
        let bytes = png()
        let conflict = StudioStore(directory: conflictRoot, importConverter: { request in
            try Data().write(to: root.appendingPathComponent(request.title + ".converted"))
            return bytes
        })
        conflict.question = "Local draft before conflict"
        let externalBytes = try JSONEncoder().encode(external)
        try externalBytes.write(to: conflictRoot.appendingPathComponent("library.json"), options: .atomic)
        conflict.importImages(["batch-one", "batch-two", "batch-three"].map { root.appendingPathComponent($0 + ".png") })
        try await waitForImport(conflict)
        verify(!FileManager.default.fileExists(atPath: root.appendingPathComponent("batch-two.converted").path), "Library conflict must stop the batch before decoding more queued images")
        verify(conflict.hasUnsavedChanges && conflict.hasLibraryConflict && !conflict.canImport && conflict.canReloadLibrary, "Import conflict must expose recovery immediately and block new imports")
        verify(try Data(contentsOf: conflictRoot.appendingPathComponent("library.json")) == externalBytes, "A conflicted import must preserve the external library byte-for-byte")
        verify((try FileManager.default.contentsOfDirectory(atPath: conflictRoot.appendingPathComponent("Images").path)).isEmpty, "A conflicted import must not leave orphan image copies")
        conflict.reloadLibrary()
        let copies = try FileManager.default.contentsOfDirectory(at: conflictRoot.appendingPathComponent("Recovery"), includingPropertiesForKeys: nil)
        let backup = try JSONDecoder().decode(LibraryState.self, from: Data(contentsOf: copies[0]))
        verify(backup.drafts[first.id.uuidString] == "Local draft before conflict" && conflict.canImport, "Conflict recovery must back up the local draft and re-enable import")
        conflict.importImages([source])
        try await waitForImport(conflict)
        verify(conflict.sessions.count == 2 && conflict.selected?.title == "source" && !conflict.hasUnsavedChanges,
               "A fresh import must complete normally after conflict recovery")
        verify(!FileManager.default.fileExists(atPath: root.appendingPathComponent("batch-three.converted").path),
               "Conflict recovery must not silently resume the discarded batch tail")
        conflict.shutdown()

        var reportImage = first; reportImage.title = "Title\r\ncontinued\rfinal"
        reportImage.questions = [QuestionRecord(question: "line one\r\nline two\rline three", answers: [Prediction(answer: "one\r\ntwo\rthree|pipe", score: 0.5)], elapsed: 0, model: "fixture", checkpoint: "fixture", device: "cpu")]
        let report = ConversationExporter.markdown(reportImage)
        verify(!report.contains("\r"), "Readable export must normalize Windows and classic Mac line endings")
        verify(report.contains("> line one\n> line two\n> line three") && report.contains("| one two three\\|pipe |"), "Multiline questions and answers must retain valid Markdown blocks and table rows")
        fflush(stdout)
        precondition(failures.isEmpty, "\(failures.count) recovery regressions failed")
    }

    private static func waitForImport(_ store: StudioStore) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while store.importing && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(!store.importing, "The isolated import must finish within its test deadline")
    }

    private static func integrityChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let storage = LibraryStorage(directory: root)
        let session = ImageSession(title: "Fixture", imageName: "owned.png")
        var state = LibraryState(sessions: [session], selectedID: session.id)
        state.sessions.append(session)
        // Bypass save validation to simulate a damaged existing file, never a user's library.
        let invalid = try JSONEncoder().encode(state)
        try invalid.write(to: root.appendingPathComponent("library.json"))
        rejects({ _ = try storage.load() }, "Duplicate image IDs must be rejected before reaching SwiftUI")
        let protected = StudioStore(directory: root)
        protected.shutdown()
        let protectedBytes = try Data(contentsOf: root.appendingPathComponent("library.json"))
        check(!protected.canModifyLibrary && protectedBytes == invalid, "Duplicate-ID libraries must stay protected byte-for-byte")
        check(protected.libraryProblem != nil, "The persistent recovery state must explain read-only mode")
        protected.reloadLibrary()
        check(!protected.canModifyLibrary, "Retrying an invalid library must not enable destructive writes")
        try FileManager.default.removeItem(at: root.appendingPathComponent("library.json"))
        protected.reloadLibrary()
        check(!protected.canModifyLibrary, "A missing repaired file must not silently become an empty writable library")
        try JSONEncoder().encode([session]).write(to: root.appendingPathComponent("history.json"))
        protected.reloadLibrary()
        check(!protected.canModifyLibrary, "Repairing a new library must not silently fall back to stale legacy history")
        var repaired = LibraryState(sessions: [session], selectedID: session.id)
        repaired.drafts = [session.id.uuidString: "Restored draft"]
        try storage.save(repaired)
        protected.reloadLibrary()
        check(protected.canModifyLibrary && protected.libraryProblem == nil, "A repaired library must reload without restarting the app")
        check(protected.selected?.id == session.id && protected.question == "Restored draft", "Reload must restore actual selection and drafts")
        protected.shutdown()
        state.sessions = [session]; state.archived = [ArchivedImage(session: session)]
        rejects({ try storage.save(state) }, "Active and archived images cannot share an ID")
        state.archived = []; state.sessions[0].imageName = "../outside.png"
        rejects({ try storage.save(state) }, "Owned image paths must stay inside Images")
        state.sessions[0] = session
        let question = QuestionRecord(question: "fixture", answers: [], elapsed: 0, model: "fixture", checkpoint: "fixture", device: "cpu")
        state.sessions[0].questions = [question, question]
        rejects({ try storage.save(state) }, "Duplicate question IDs must not reach a lazy answer list")
        state.sessions[0].questions = []; state.version = 2
        rejects({ try storage.save(state) }, "Unsupported library versions must not be written as current format")
        state.version = 1
        let answer = Prediction(answer: "duplicate", score: 0.5)
        state.sessions[0].questions = [QuestionRecord(question: "fixture", answers: [answer, answer], elapsed: 0, model: "fixture", checkpoint: "fixture", device: "cpu")]
        rejects({ try storage.save(state) }, "Duplicate answer identities must not reach SwiftUI's ForEach")
        state.sessions[0].questions = [QuestionRecord(question: "fixture", answers: [Prediction(answer: "invalid", score: 2)], elapsed: 0, model: "fixture", checkpoint: "fixture", device: "cpu")]
        rejects({ try storage.save(state) }, "Persisted candidate scores must remain in the probability range")
        state.sessions[0].questions = [QuestionRecord(question: "fixture", answers: [], elapsed: -1, model: "fixture", checkpoint: "fixture", device: "cpu")]
        rejects({ try storage.save(state) }, "Negative elapsed times must be rejected as damaged data")
    }

    private static func authorityChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let old = ImageSession(title: "Legacy", imageName: "legacy.png")
        try JSONEncoder().encode([old]).write(to: root.appendingPathComponent("history.json"))
        let storage = LibraryStorage(directory: root)
        var current = try storage.load()
        current.sessions.append(ImageSession(title: "Newer saved image", imageName: "newer.png"))
        try storage.save(current)
        let snapshot = root.appendingPathComponent("library.json")
        let saved = root.appendingPathComponent("retained-library.json")
        try FileManager.default.moveItem(at: snapshot, to: saved)
        let restarted = StudioStore(directory: root)
        check(!restarted.canModifyLibrary && restarted.libraryProblem != nil, "A missing authoritative library after restart must not silently migrate stale legacy history")
        restarted.shutdown()

        let emptyRoot = root.appendingPathComponent("without-legacy")
        try LibraryStorage(directory: emptyRoot).save(LibraryState())
        try FileManager.default.moveItem(at: emptyRoot.appendingPathComponent("library.json"), to: emptyRoot.appendingPathComponent("retained.json"))
        let emptyRestart = StudioStore(directory: emptyRoot)
        check(!emptyRestart.canModifyLibrary, "A missing established primary without legacy history must not become a new writable empty library")
        emptyRestart.shutdown()

        let blocked = root.appendingPathComponent("marker-failure")
        try FileManager.default.createDirectory(at: blocked, withIntermediateDirectories: true)
        let primary = blocked.appendingPathComponent("library.json")
        let original = try JSONEncoder().encode(current)
        try original.write(to: primary)
        let preexisting = LibraryStorage(directory: blocked)
        _ = try preexisting.load()
        try FileManager.default.createDirectory(at: blocked.appendingPathComponent(".library-established"), withIntermediateDirectories: false)
        var changed = current; changed.sessions[0].title = "Uncommitted edit"
        rejects({ try preexisting.save(changed) }, "An authority marker failure must be reported before committing a new snapshot")
        let unchanged = try Data(contentsOf: primary)
        check(unchanged == original, "A failed save must never commit the library and then trigger image rollback")
        check(!FileManager.default.fileExists(atPath: snapshot.path), "Protected restart must not recreate a stale library at the old path")
        try FileManager.default.moveItem(at: saved, to: snapshot)
        restarted.reloadLibrary()
        check(restarted.canModifyLibrary && restarted.sessions.count == 2, "Restoring the actual primary file must recover all newer records")
        restarted.shutdown()
    }

    private static func conflictChecks(_ root: URL) throws {
        let session = ImageSession(title: "Original", imageName: "owned.png")
        let initial = LibraryState(sessions: [session], selectedID: session.id)
        let storage = LibraryStorage(directory: root)
        try storage.save(initial)
        let snapshot = root.appendingPathComponent("library.json")
        let original = try Data(contentsOf: snapshot)
        let store = StudioStore(directory: root)
        store.question = "Unsaved local draft"
        var external = initial; external.sessions[0].title = "Restored in another app"
        let externalBytes = try JSONEncoder().encode(external)
        try externalBytes.write(to: snapshot, options: .atomic)
        store.retrySave()
        let retained = try Data(contentsOf: snapshot)
        check(retained == externalBytes && store.hasLibraryConflict && store.hasUnsavedChanges, "Autosave must not overwrite a library changed after startup")
        check(store.question == "Unsaved local draft", "Detecting a conflict must retain the in-memory draft")
        store.busy = true; store.reloadLibrary()
        check(store.hasLibraryConflict, "Recovery must not replace the library underneath an active request")
        store.busy = false; store.reloadLibrary()
        check(!store.hasLibraryConflict && !store.hasUnsavedChanges && store.selected?.title == external.sessions[0].title, "Explicit recovery must load the external version after backing up local changes")
        let copies = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("Recovery"), includingPropertiesForKeys: nil)
        let backup = try JSONDecoder().decode(LibraryState.self, from: Data(contentsOf: copies.first!))
        check(backup.drafts[session.id.uuidString] == "Unsaved local draft" && backup.sessions == initial.sessions, "Recovery copies must contain the full local state before any reload")
        store.question = "A draft on the recovered version"; store.retrySave()
        check(!store.hasUnsavedChanges, "Saving must resume against the newly loaded revision")
        store.shutdown()

        let missing = StudioStore(directory: root)
        try FileManager.default.removeItem(at: snapshot)
        missing.question = "Do not recreate a moved library"; missing.retrySave()
        check(missing.hasLibraryConflict && !FileManager.default.fileExists(atPath: snapshot.path), "A moved library must not silently be recreated over its old path")
        missing.reloadLibrary()
        check(missing.hasLibraryConflict && missing.question == "Do not recreate a moved library", "A missing replacement must leave current edits recoverable")
        try original.write(to: snapshot)
        missing.reloadLibrary(); missing.shutdown()

        let corrupt = root.appendingPathComponent("corrupt")
        try FileManager.default.createDirectory(at: corrupt, withIntermediateDirectories: true)
        let bad = Data("{damaged".utf8)
        try bad.write(to: corrupt.appendingPathComponent("library.json"))
        rejects({ try LibraryStorage(directory: corrupt).save(initial) }, "A first save must not overwrite an existing corrupt library without loading it")
        let preserved = try Data(contentsOf: corrupt.appendingPathComponent("library.json"))
        check(preserved == bad, "A rejected first save must preserve corrupt source bytes")

        let linked = root.appendingPathComponent("dangling")
        try FileManager.default.createDirectory(at: linked, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: linked.appendingPathComponent("library.json"), withDestinationURL: linked.appendingPathComponent("missing-target"))
        rejects({ _ = try LibraryStorage(directory: linked).load() }, "A broken symbolic link must not be treated as a brand-new empty library")

        let migrated = root.appendingPathComponent("legacy-conflict")
        try FileManager.default.createDirectory(at: migrated, withIntermediateDirectories: true)
        let legacy = migrated.appendingPathComponent("history.json")
        try JSONEncoder().encode([session]).write(to: legacy)
        let migration = LibraryStorage(directory: migrated)
        _ = try migration.load()
        try JSONEncoder().encode([ImageSession(title: "External", imageName: "other.png")]).write(to: legacy)
        rejects({ try migration.save(initial) }, "Legacy history changed during migration must not be replaced with a stale snapshot")
    }

    private static func settingsAndExportChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root.appendingPathComponent("vqa_project"), withIntermediateDirectories: true)
        for name in ["vqa_project/desktop_worker.py", "python", "config.yaml", "weights.pt"] {
            try Data("filesystem fixture only".utf8).write(to: root.appendingPathComponent(name))
        }
        let python = root.appendingPathComponent("python")
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: python.path)
        var settings = AppSettings(project: "  \(root.path)\n", python: " \(python.path)\n", config: " config.yaml\n", checkpoint: " weights.pt ")
        check(settings.validationMessage == nil, "Pasted paths must be validated after whitespace normalization")
        check(settings.absolute(" weights.pt\n") == root.appendingPathComponent("weights.pt").path, "Runtime paths must match normalized settings validation")
        settings.topk = 0
        check(settings.validationMessage != nil, "Out-of-range saved candidate counts must be rejected")
        settings.topk = 5; settings.device = "unknown"
        check(settings.validationMessage != nil, "Unsupported saved device settings must be explained")
        let session = ImageSession(title: "birds / café:\nnear trees", imageName: "fixture.png")
        let filename = ConversationExporter.filename(for: session, readable: true)
        check(!filename.contains("/") && !filename.contains(":") && !filename.contains("\n") && filename.hasSuffix(".md"), "Renamed display titles must produce a safe default export filename")
        let longTitle = ImageSession(title: String(repeating: "图片", count: 150), imageName: "fixture.png")
        check(ConversationExporter.filename(for: longTitle, readable: false).utf8.count < 255, "Long Chinese titles must not exceed the filesystem's filename byte limit")
        check(LibraryQuery.results([session], query: "cafe", favoritesOnly: false, favorites: [], sort: .name).count == 1, "Library search should ignore diacritics like native search")
        let deadline = RequestDeadline()
        check(!deadline.finish() && !deadline.expire(), "A finished request's timer must not expire a subsequent request")
        let expired = RequestDeadline()
        check(expired.expire() && !expired.expire() && expired.finish(), "A request deadline must be claimed only once and retain timeout status")
    }

    private static func fileOpeningChecks(_ root: URL) {
        let delegate = StudioAppDelegate()
        let first = root.appendingPathComponent("first.png")
        let second = root.appendingPathComponent("second.png")
        delegate.receiveImages([first])
        delegate.receiveImages([URL(string: "https://example.invalid/not-an-import")!, second])
        var deliveries: [[URL]] = []
        delegate.importImages = { deliveries.append($0) }
        check(deliveries == [[first, second]], "File-open events arriving before the workspace exists must be queued in order")
        delegate.importImages = { deliveries.append($0) }
        check(deliveries.count == 1, "Reopening a window must not import its previously delivered files again")
        delegate.receiveImages([second])
        check(deliveries == [[first, second], [second]], "A new file-open event must reach the existing workspace once")
    }

    private static func png() -> Data {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 3, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                      colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 255, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        return bitmap.representation(using: .png, properties: [:])!
    }

    private static func persistenceChecks(_ root: URL, settings: AppSettings) async throws {
        let directory = root.appendingPathComponent("library")
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("Images"), withIntermediateDirectories: true)
        let first = ImageSession(title: "First", imageName: "first.png")
        let second = ImageSession(title: "Second", imageName: "second.png")
        let storage = LibraryStorage(directory: directory)
        try storage.save(LibraryState(sessions: [first, second], selectedID: first.id))
        try png().write(to: directory.appendingPathComponent("Images/first.png"))
        let store = StudioStore(directory: directory)
        store.imageZoom = .quadruple; store.selectedID = second.id
        check(store.imageZoom == .fit, "Switching pictures must not carry an extreme crop into the next picture")
        store.selectedID = first.id; store.settings = settings; store.ready = true
        await store.preview.waitUntilLoaded()
        store.question = "stubborn"
        let waiting = Task { await store.ask() }
        let marker = URL(fileURLWithPath: settings.project).appendingPathComponent("stubborn.pid")
        try await waitFor(marker)
        let immediate = try storage.load()
        check(immediate.drafts[first.id.uuidString] == "stubborn", "Submitted text must be durable before waiting for the worker")
        store.question = "A newer draft"
        try await Task.sleep(for: .milliseconds(450))
        let duringRequest = try LibraryStorage(directory: directory).load()
        check(duringRequest.drafts[first.id.uuidString] == "A newer draft" && duringRequest.interruptedQuestions?.contains(where: { $0.question == "stubborn" }) == true, "A crash during inference must preserve both submitted text and newer typing")
        store.cancelModelTask()
        await waiting.value
        check(!store.busy && !store.ready && store.question == "A newer draft", "Cancelled inference must preserve newer typing and become reloadable")
        check(store.selected?.questions.isEmpty == true, "Cancelled predictions must not invent answer history")
        let afterCancellation = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("library.json"))) as! [String: Any]
        let recovered = afterCancellation["interruptedQuestions"] as? [[String: Any]] ?? []
        check(recovered.contains { $0["question"] as? String == "stubborn" }, "Cancelling after typing a newer draft must retain the submitted question separately")

        store.question = "abort"; store.ready = true
        await store.ask()
        check(!store.ready && !store.busy && store.question == "abort", "Unexpected worker exit must restore the draft and clear ready status")
        store.errorMessage = nil

        // A directory at the snapshot filename simulates failed atomic replacement even when run as root.
        let snapshot = directory.appendingPathComponent("library.json")
        let beforeSaveFailure = try Data(contentsOf: snapshot)
        try FileManager.default.removeItem(at: snapshot)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        store.question = "Keep this after a failed quit"
        let quitFailed = await store.prepareToTerminate()
        check(!quitFailed && !store.quitting && store.hasUnsavedChanges, "Save failure must cancel quit and leave recoverable in-memory edits")
        check(store.question == "Keep this after a failed quit", "A failed quit must not clear the draft")
        store.errorMessage = nil; store.question += "!"
        try await Task.sleep(for: .milliseconds(450))
        check(store.errorMessage == nil && store.hasUnsavedChanges, "Ongoing failed autosaves should retain the banner without repeatedly opening alerts")
        try FileManager.default.removeItem(at: snapshot)
        try beforeSaveFailure.write(to: snapshot)
        store.retrySave()
        let retried = try storage.load()
        check(!store.hasUnsavedChanges && retried.drafts[first.id.uuidString] == store.question, "Retry Save must persist the actual latest draft")
        let quitSucceeded = await store.prepareToTerminate()
        check(quitSucceeded, "Quit must succeed after storage recovers")

        let importDirectory = root.appendingPathComponent("queued-library")
        let source = root.appendingPathComponent("source.png")
        try png().write(to: source)
        let importing = StudioStore(directory: importDirectory)
        importing.importImages([source, source, source])
        let quitAfterImport = await importing.prepareToTerminate()
        let imported = try LibraryStorage(directory: importDirectory).load()
        check(quitAfterImport && !importing.importing && imported.sessions.count == 3, "Quit must finish and save all accepted imports")
        importing.importImages([source])
        check(!importing.importing, "New imports must not enter a queue already finishing termination")
    }

    private static func unfinishedQuestionChecks(_ root: URL) throws {
        let first = ImageSession(title: "One", imageName: "one.png")
        let second = ImageSession(title: "Two", imageName: "two.png")
        let submission = PendingQuestion(sessionID: first.id, question: "Submitted before interruption")
        let other = PendingQuestion(sessionID: second.id, question: "Another image's unfinished question")
        var state = LibraryState(sessions: [first, second], drafts: [first.id.uuidString: "Newer typing", second.id.uuidString: "Other draft"], selectedID: first.id)
        state.interruptedQuestions = [submission, other]
        let storage = LibraryStorage(directory: root)
        try storage.save(state)
        let store = StudioStore(directory: root)
        check(store.question == "Newer typing" && store.recoverableQuestions == [submission], "Restart must expose interrupted work without replacing newer typing")
        store.recoverQuestion(submission.id)
        check(store.question == submission.question && store.recoverableQuestions.map(\.question) == ["Newer typing"], "Recovery must swap the existing draft into recoverable storage")
        let saved = try LibraryStorage(directory: root).load()
        check(saved.drafts[first.id.uuidString] == submission.question && saved.interruptedQuestions?.contains(where: { $0.question == "Newer typing" }) == true, "Both sides of a recovery swap must be durable immediately")
        let swapBack = store.recoverableQuestions[0]
        store.busy = true; store.recoverQuestion(swapBack.id)
        check(store.question == submission.question, "Recovery must not replace text during an active request")
        store.busy = false; store.selectedID = second.id
        store.recoverQuestion(swapBack.id)
        check(store.question == "Other draft" && store.selectedID == second.id, "A stale recovery action cannot alter a different image")
        store.selectedID = first.id; store.recoverQuestion(swapBack.id)
        check(store.question == "Newer typing" && store.recoverableQuestions[0].question == submission.question, "Repeated recovery swaps must never discard either question")
        store.remove(first.id); store.restoreArchived(first.id)
        check(store.question == "Newer typing" && store.recoverableQuestions[0].question == submission.question, "Archive and restore must retain unfinished questions with their image")
        store.question = " \n "
        store.recoverQuestion(store.recoverableQuestions[0].id)
        check(store.question == submission.question && !store.hasUnsavedChanges, "Whitespace-only drafts must not make recovery fail validation")
        store.shutdown()

        let emptyRoot = root.appendingPathComponent("empty-draft")
        var pendingOnly = LibraryState(sessions: [first], selectedID: first.id)
        pendingOnly.interruptedQuestions = [submission]
        try LibraryStorage(directory: emptyRoot).save(pendingOnly)
        let reopened = StudioStore(directory: emptyRoot)
        check(reopened.question == submission.question && reopened.recoverableQuestions.isEmpty, "An interrupted question should return directly to an empty composer")
        reopened.shutdown()
        let finishedRecovery = try LibraryStorage(directory: emptyRoot).load()
        check(finishedRecovery.interruptedQuestions == nil && finishedRecovery.drafts[first.id.uuidString] == submission.question, "Automatic restoration must not duplicate the same question on every launch")
        pendingOnly.interruptedQuestions = [submission, submission]
        rejects({ try LibraryStorage(directory: root.appendingPathComponent("duplicates")).save(pendingOnly) }, "Duplicate interrupted IDs must be rejected before building a recovery menu")
        pendingOnly.interruptedQuestions = [PendingQuestion(sessionID: UUID(), question: "Detached")]
        rejects({ try LibraryStorage(directory: root.appendingPathComponent("orphan")).save(pendingOnly) }, "Interrupted questions must belong to a retained active or archived image")
        pendingOnly.interruptedQuestions = [PendingQuestion(sessionID: first.id, question: " \n ")]
        rejects({ try LibraryStorage(directory: root.appendingPathComponent("blank")).save(pendingOnly) }, "Empty interrupted questions must not be written")

        let legacy = try JSONEncoder().encode(LibraryState(sessions: [first], selectedID: first.id))
        let decoded = try JSONDecoder().decode(LibraryState.self, from: legacy)
        check(decoded.interruptedQuestions == nil, "Version-1 libraries without interrupted work must keep decoding without migration")

        let conflictRoot = root.appendingPathComponent("recovery-conflict")
        try LibraryStorage(directory: conflictRoot).save(state)
        let conflicted = StudioStore(directory: conflictRoot)
        var external = state; external.sessions[0].title = "Externally changed"
        try JSONEncoder().encode(external).write(to: conflictRoot.appendingPathComponent("library.json"), options: .atomic)
        conflicted.recoverQuestion(submission.id)
        check(conflicted.hasLibraryConflict && conflicted.question == "Newer typing" && conflicted.recoverableQuestions == [submission], "Failed recovery saves must roll back both composer and pending list without discarding either")
        conflicted.reloadLibrary()
        check(conflicted.canModifyLibrary && !conflicted.hasLibraryConflict && conflicted.question == "Newer typing", "A failed recovery swap must remain recoverable through the conflict backup workflow")
        conflicted.shutdown()
    }

    private static func settingsApplicationChecks(_ root: URL, settings: AppSettings) async throws {
        let priorDefaults = UserDefaults.standard.data(forKey: "modelSettings")
        defer {
            if let priorDefaults { UserDefaults.standard.set(priorDefaults, forKey: "modelSettings") }
            else { UserDefaults.standard.removeObject(forKey: "modelSettings") }
        }
        let store = StudioStore(directory: root)
        defer { store.shutdown() }
        store.settings = settings
        await store.loadModel()
        check(store.ready, "The protocol-only model must be ready before testing settings application")
        let requestFile = URL(fileURLWithPath: settings.project).appendingPathComponent("last-request.json")
        let loaded = try Data(contentsOf: requestFile)
        var updated = settings; updated.topk = 3
        check(!store.needsModelReload(for: updated), "Changing only candidate count must not require reloading weights")
        await store.apply(updated)
        let after = try Data(contentsOf: requestFile)
        check(store.ready && !store.busy && store.settings.topk == 3 && after == loaded, "Applying candidate count must keep the already running worker without a load request")
        let pidFile = requestFile.deletingLastPathComponent().appendingPathComponent("worker.pid")
        let originalPID = try Data(contentsOf: pidFile)
        let defaultsAfterUpdate = UserDefaults.standard.data(forKey: "modelSettings")
        let invalidConfig = root.appendingPathComponent("blocked-config.yaml")
        precondition(mkfifo(invalidConfig.path, 0o600) == 0)
        var invalid = updated; invalid.config = invalidConfig.path
        let issue = await store.apply(invalid)
        let sameRequest = try Data(contentsOf: requestFile) == loaded
        check(store.ready && !store.busy && store.settings == updated && sameRequest,
              "Rejecting a pipe must leave the current settings and loaded worker usable")
        check(issue?.contains("配置") == true && UserDefaults.standard.data(forKey: "modelSettings") == defaultsAfterUpdate,
              "Rejected input must explain the invalid field and leave the saved settings intact")
        check(store.needsModelReload(for: updated), "Explicit Reload with unchanged settings must still reload")
        var different = updated; different.offline.toggle()
        check(store.needsModelReload(for: different), "Changing model resource policy must still require a fresh model")
        let first = ImageSession(title: "Top-k check", imageName: "input.png")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Images"), withIntermediateDirectories: true)
        try png().write(to: root.appendingPathComponent("Images/input.png"))
        store.sessions = [first]; store.selectedID = first.id
        await store.preview.waitUntilLoaded()
        store.question = "topk-check"
        await store.ask()
        check(store.selected?.questions.last?.answers.count == 3, "The next prediction must use the updated candidate count")
        let sameWorker = try Data(contentsOf: pidFile) == originalPID
        check(sameWorker, "The next prediction must still use the same worker after rejected settings")
    }

    private static func bootstrapChecks(_ root: URL, settings: AppSettings) async throws {
        try FileManager.default.copyItem(at: URL(fileURLWithPath: settings.project), to: root)
        var spaced = settings; spaced.project = root.path
        let client = BackendClient(logDirectory: root.appendingPathComponent("Local Support"))
        defer { client.shutdown() }
        let response = try await client.request(settings: spaced, action: "load")
        check(response.ok, "The local bootstrap must load projects with spaces and Chinese characters without shell quoting")
        let info = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("startup-info.json"))) as! [String: String]
        let cwd = URL(fileURLWithPath: info["cwd"] ?? "/").resolvingSymlinksInPath()
        let restoredPaths = cwd.path == root.resolvingSymlinksInPath().path && info["import_root"] == root.path
        if !restoredPaths { print("Bootstrap paths: \(info); expected \(root.path), canonical \(root.resolvingSymlinksInPath().path)"); fflush(stdout) }
        check(restoredPaths, "The bootstrap must restore the project cwd and absolute import root, including the macOS /var alias")
        check(info["relative_config"] == "PROTOCOL_TEST_ONLY; not model data", "Project-relative resources must retain their original meaning after local initialization")
    }

    private static func startupLifetimeChecks(_ root: URL, settings: AppSettings) async throws {
        let project = URL(fileURLWithPath: settings.project)
        let hold = project.appendingPathComponent("hold-startup")
        let entered = project.appendingPathComponent("startup-entered")
        let lastRequest = project.appendingPathComponent("last-request.json")
        try Data().write(to: hold)
        try? FileManager.default.removeItem(at: entered)
        defer { try? FileManager.default.removeItem(at: hold) }
        var slow = settings; slow.config = "held-load.yaml"
        let store = StudioStore(directory: root.appendingPathComponent("window-independent"))
        store.settings = slow
        let windowLifetime = Task { store.beginStartup() }
        await windowLifetime.value
        try await waitFor(entered)
        windowLifetime.cancel()
        try FileManager.default.removeItem(at: hold)
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while store.busy && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(store.ready && !store.busy, "Finishing or cancelling the window task must not cancel application-owned model startup")
        let loaded = try Data(contentsOf: lastRequest)
        store.beginStartup()
        await Task.yield()
        let stillLoaded = try Data(contentsOf: lastRequest)
        check(stillLoaded == loaded, "Reopening a workspace must not launch another startup request")
        store.shutdown()

        try Data().write(to: hold)
        try FileManager.default.removeItem(at: entered)
        let stopped = StudioStore(directory: root.appendingPathComponent("explicit-stop"))
        stopped.settings = slow; stopped.beginStartup()
        try await waitFor(entered)
        stopped.cancelModelTask()
        let stopDeadline = ContinuousClock.now.advanced(by: .seconds(4))
        while stopped.busy && ContinuousClock.now < stopDeadline { try await Task.sleep(for: .milliseconds(10)) }
        check(!stopped.busy && !stopped.ready && stopped.status == "模型已停止", "Explicit Stop must still cancel application-owned startup and leave a reloadable model")
        try FileManager.default.removeItem(at: hold)
        stopped.settings = settings
        await stopped.loadModel()
        check(stopped.ready, "A user-stopped initial load must support a normal manual reload")
        stopped.shutdown()
    }

    private static func waitFor(_ marker: URL) async throws {
        let limit = ContinuousClock.now.advanced(by: .seconds(4))
        while !FileManager.default.fileExists(atPath: marker.path) && ContinuousClock.now < limit {
            try await Task.sleep(for: .milliseconds(10))
        }
        check(FileManager.default.fileExists(atPath: marker.path), "The test worker must have entered its intended operation")
    }

    private static func transportChecks(_ root: URL, settings: AppSettings) async throws {
        let client = BackendClient(logDirectory: root.appendingPathComponent("reliability-logs"))
        defer { client.shutdown() }
        for action in ["partial", "malformed", "abort"] {
            do { _ = try await client.request(settings: settings, action: action); preconditionFailure("Invalid response must be rejected: \(action)") }
            catch let error as BackendError { check(error.needsReload, "Invalid transport responses must require reload") }
            let recovery = try await client.request(settings: settings, action: "load")
            check(recovery.ok, "A transport failure must clean up its process for the next request")
        }
        for question in ["score-outside-range", "duplicate-answers", "negative-elapsed"] {
            do { _ = try await client.request(settings: settings, action: "predict", question: question); preconditionFailure("Invalid predictions must not enter history") }
            catch let error as BackendError { check(error.needsReload, "Invalid answer data must require reload") }
        }
        do { _ = try await client.request(settings: settings, action: "rejected"); preconditionFailure("Worker rejection must be surfaced") }
        catch let error as BackendError { check(!error.needsReload, "A valid model rejection must remain distinct from broken transport") }
        let stillAlive = try await client.request(settings: settings, action: "load")
        check(stillAlive.ok, "A valid rejection must not poison subsequent requests")

        let marker = root.appendingPathComponent("stubborn.pid")
        try? FileManager.default.removeItem(at: marker)
        let task = Task { try await client.request(settings: settings, action: "stubborn") }
        try await waitFor(marker)
        let pid = Int32(try String(contentsOf: marker, encoding: .utf8))!
        let start = ContinuousClock.now
        task.cancel()
        do { _ = try await task.value; preconditionFailure("Swift Task cancellation must stop the request") }
        catch is CancellationError { checks += 1 }
        check(start.duration(to: .now) < .seconds(3), "A SIGTERM-ignoring worker must still be cancelled promptly")
        let exitLimit = ContinuousClock.now.advanced(by: .seconds(2))
        while kill(pid, 0) == 0 && ContinuousClock.now < exitLimit { try await Task.sleep(for: .milliseconds(10)) }
        check(kill(pid, 0) != 0, "A stubborn cancelled worker must not remain alive")
        let recovered = try await client.request(settings: settings, action: "load")
        check(recovered.ok, "The transport must restart after escalated cancellation")

        let timeoutClient = BackendClient(logDirectory: root.appendingPathComponent("timeout-logs"), requestTimeout: 0.15)
        defer { timeoutClient.shutdown() }
        let timeoutStart = ContinuousClock.now
        do { _ = try await timeoutClient.request(settings: settings, action: "stubborn"); preconditionFailure("A timed-out request must not finish as successful") }
        catch { check(error.localizedDescription.contains("超时"), "Timeouts must explain the actual failure") }
        check(timeoutStart.duration(to: .now) < .seconds(3), "Timeouts must terminate workers that ignore SIGTERM")
        let afterTimeout = try await timeoutClient.request(settings: settings, action: "load")
        check(afterTimeout.ok, "The next request must recover after timeout cleanup")
    }
}
