import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

@MainActor @main
enum ImageWorkflowChecks {
    private static var checks = 0
    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message); checks += 1
    }

    static func main() async throws {
        if ImageImportWorker.runIfRequested() { return }
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent("image-workflows")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try await codecAndCacheChecks(root.appendingPathComponent("codec"))
        try stampChecks(root.appendingPathComponent("stamps"))
        try snapshotChecks(root.appendingPathComponent("snapshots"))
        try await previewChecks(root.appendingPathComponent("preview"))
        try await draftChecks(root.appendingPathComponent("drafts"))
        try await clipboardChecks(root.appendingPathComponent("clipboard"))
        try await importControlChecks(root.appendingPathComponent("import-control"))
        try await importRollbackChecks(root.appendingPathComponent("import-rollback"))
        try await quitErrorPresentationChecks(root.appendingPathComponent("quit-errors"))
        try await existingErrorQuitChecks(root.appendingPathComponent("existing-error-quit"))
        errorPresentationChecks()
        try await saveErrorIdentityChecks(root.appendingPathComponent("save-error-identity"))
        try await feedbackChecks()
        conversationChecks()
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) image/interaction regression checks passed (isolated generated image fixtures; no model inference)")
    }

    private static func fixture(_ url: URL, width: Int = 2, height: Int = 3, orientation: Int? = nil) throws {
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                      colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 180, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let orientation {
            let output = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)!
            CGImageDestinationAddImage(output, bitmap.cgImage!, [kCGImagePropertyOrientation: orientation] as CFDictionary)
            precondition(CGImageDestinationFinalize(output))
        } else { try bitmap.representation(using: .png, properties: [:])!.write(to: url, options: .atomic) }
    }

    private static func codecAndCacheChecks(_ root: URL) async throws {
        let first = root.appendingPathComponent("first.png")
        try fixture(first)
        let source = try Data(contentsOf: first)
        let image = try ImageCodec.decode(first)
        check(image.width == 2 && image.height == 3, "Preview must retain real pixel dimensions")
        let converted = try ImageImport.pngData(at: first)
        let imported = root.appendingPathComponent("imported.png")
        try converted.write(to: imported)
        let normalized = try ImageCodec.decode(imported)
        check(normalized.width == 2 && normalized.height == 3, "Normalized imports must remain decodable at the same dimensions")
        let after = try Data(contentsOf: first)
        check(source == after, "Preview/import conversion must never change the source image")
        let rotated = root.appendingPathComponent("rotated.jpg")
        try fixture(rotated, orientation: 6)
        let oriented = try ImageCodec.decode(rotated)
        check(oriented.width == 3 && oriented.height == 2, "EXIF orientation must be applied to the actual rendered pixels")
        let orientedPNG = root.appendingPathComponent("rotated.png")
        try ImageImport.pngData(at: rotated).write(to: orientedPNG)
        let roundTrip = try ImageCodec.decode(orientedPNG)
        check(roundTrip.width == 3 && roundTrip.height == 2, "PNG imports must normalize orientation once, without rotating again on reopen")
        let link = root.appendingPathComponent("linked.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: first)
        let linked = try ImageCodec.decode(link)
        check(linked.width == 2 && linked.height == 3, "Valid image symlinks must remain importable")
        let cache = StudioImageCache.shared
        let thumbnail = await cache.thumbnail(at: first)!
        let reused = await cache.thumbnail(at: first)
        check(reused === thumbnail, "Unchanged thumbnails must reuse their bounded cache entry")
        try fixture(first, width: 5, height: 4)
        let changed = await cache.thumbnail(at: first)
        check(changed != nil && changed !== thumbnail, "Replacing a file at the same path must invalidate its thumbnail")
        check(changed!.size.width > changed!.size.height && max(changed!.size.width, changed!.size.height) <= 84, "Invalidated thumbnails must reflect the replacement aspect ratio within the cache bound")
        try FileManager.default.removeItem(at: first)
        let deleted = await cache.thumbnail(at: first)
        check(deleted == nil, "Deleting a file must not leave a stale thumbnail visible")
        let corrupt = root.appendingPathComponent("corrupt.png")
        try Data("not an image".utf8).write(to: corrupt)
        do { _ = try ImageCodec.decode(corrupt); preconditionFailure("A readable corrupt file must not count as a valid image") }
        catch { checks += 1 }
    }

    private static func stampChecks(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("image.png")
        let bytes = Data([1, 2, 3, 4])
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        try bytes.write(to: file, options: .atomic)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: file.path)
        let original = try ImageFileStamp.read(file)
        check(original.size == 4 && original.modified == date, "Stamps must retain regular-file size and modification time")
        let link = root.appendingPathComponent("link.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
        let linked = try ImageFileStamp.read(link)
        check(linked == original, "Stamps must follow the image target rather than the symbolic link")
        let replacement = root.appendingPathComponent("replacement.png")
        try bytes.write(to: replacement, options: .atomic)
        try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: replacement.path)
        try FileManager.default.removeItem(at: file)
        try FileManager.default.moveItem(at: replacement, to: file)
        let changed = try ImageFileStamp.read(file)
        check(changed.size == original.size && changed.modified == original.modified && changed.fileNumber != original.fileNumber,
              "Atomic replacement must invalidate the stamp even with identical size and modification time")
        let followed = try ImageFileStamp.read(link)
        check(followed == changed, "A symbolic link must observe a replaced target")
        do { _ = try ImageFileStamp.read(root); preconditionFailure("Directories must not be accepted as image files") }
        catch { checks += 1 }
        try FileManager.default.removeItem(at: file)
        for missing in [file, link] {
            do { _ = try ImageFileStamp.read(missing); preconditionFailure("Missing files and dangling links must fail") }
            catch { checks += 1 }
        }
    }

    private static func previewChecks(_ root: URL) async throws {
        let first = ImageSession(title: "First", imageName: "first.png")
        let second = ImageSession(title: "Second", imageName: "second.png")
        let broken = ImageSession(title: "Broken", imageName: "broken.png")
        let folder = root.appendingPathComponent("Images")
        try fixture(folder.appendingPathComponent(first.imageName))
        try fixture(folder.appendingPathComponent(second.imageName), width: 5, height: 4)
        try Data("readable but invalid".utf8).write(to: folder.appendingPathComponent(broken.imageName))
        try LibraryStorage(directory: root).save(LibraryState(sessions: [first, second, broken], selectedID: first.id))
        let store = StudioStore(directory: root)
        defer { store.shutdown() }
        check(store.preview.loading, "Selecting a persisted image must show an explicit loading phase")
        await store.preview.waitUntilLoaded()
        check(store.preview.image?.width == 2 && store.preview.problem == nil, "A persisted selection must decode without requiring UI body reads")
        store.ready = true; store.question = "A draft"
        check(store.canAsk, "A decoded readable image and ready model must enable asking")
        let original = store.preview.image!.bitmap
        store.question = "Keep typing"; store.search = "query"; store.imageZoom = .double
        check(store.preview.image!.bitmap === original && !store.preview.loading, "Typing/filtering/zooming must not read or decode the preview again")
        store.refreshImagePreview()
        check(store.preview.image!.bitmap === original && !store.preview.loading, "Activation must retain an unchanged decoded image")
        try fixture(folder.appendingPathComponent(first.imageName), width: 9, height: 8)
        await store.ask()
        await store.preview.waitUntilLoaded()
        check(store.question == "Keep typing" && store.selected?.questions.isEmpty == true, "Replacing a previewed file must preserve the question without invoking inference on unseen pixels")
        check(store.ready && !store.busy && store.pendingQuestion == nil, "Image preparation failure must retain the loaded model and clear pending UI")
        check(store.preview.image?.width == 9 && store.errorMessage?.contains("图片文件已发生变化") == true, "A changed input must refresh the preview and explain why resubmission is needed")
        store.selectedID = second.id
        check(store.preview.image == nil && store.preview.loading, "Selection must clear old image pixels before a new decode completes")
        store.selectedID = first.id; store.selectedID = second.id
        await store.preview.waitUntilLoaded()
        check(store.preview.url == store.imageURL && store.preview.image?.width == 5, "Rapid selection must never display the result of an obsolete decode")
        store.selectedID = broken.id; store.question = "Do not send a broken image"
        await store.preview.waitUntilLoaded()
        check(store.preview.problem != nil && !store.canAsk, "Readable corrupt images must disable model submission")
        let draft = store.question
        try fixture(folder.appendingPathComponent(broken.imageName), width: 7, height: 6)
        store.rereadImage()
        await store.preview.waitUntilLoaded()
        check(store.preview.image?.width == 7 && store.preview.problem == nil && store.canAsk, "A repaired image must become usable after rereading")
        check(store.question == draft && store.selectedID == broken.id, "Rereading must preserve selection and its draft")
        try FileManager.default.removeItem(at: folder.appendingPathComponent(broken.imageName))
        store.refreshImagePreview()
        await store.preview.waitUntilLoaded()
        check(store.preview.image == nil && store.preview.problem != nil && !store.canAsk, "Missing images must invalidate the active preview and disable submission")
        store.selectedID = nil
        check(store.preview.image == nil && store.preview.url == nil && !store.preview.loading, "Deselecting must clear preview state")
        store.selectedID = first.id
        await store.preview.waitUntilLoaded()
        check(store.preview.image?.width == 9, "A rapid deselect/reselect must recover a usable preview")
        store.question = "Keep this if the source disappears"
        try FileManager.default.removeItem(at: folder.appendingPathComponent(first.imageName))
        await store.ask(); await store.preview.waitUntilLoaded()
        check(store.question == "Keep this if the source disappears" && store.selected?.questions.isEmpty == true && store.ready,
              "Deletion after the last preview check must restore the question before any model request and retain the model")
        check(store.preview.image == nil && !store.canAsk && store.preview.problem != nil,
              "A disappeared source must settle into a disabled retryable preview state without synchronous UI file probes")
    }

    private static func snapshotChecks(_ root: URL) throws {
        let source = root.appendingPathComponent("source.png")
        try fixture(source, width: 4, height: 3)
        let stamp = try ImageFileStamp.read(source)
        let original = try Data(contentsOf: source)
        let snapshot = try PredictionImage(source: source, stamp: stamp)
        let captured = try Data(contentsOf: snapshot.url)
        check(snapshot.url != source && captured == original, "Inference must use an independent byte-identical private input")
        try fixture(source, width: 8, height: 7)
        let fixed = try ImageCodec.decode(snapshot.url)
        check(fixed.width == 4 && fixed.height == 3, "Replacing the library file must not change an already submitted worker input")
        do { _ = try PredictionImage(source: source, stamp: stamp); preconditionFailure("A stale preview must not be submitted") }
        catch is ImageSnapshotError { checks += 1 }
        snapshot.remove()
        check(!FileManager.default.fileExists(atPath: snapshot.url.path), "A completed or failed request must be able to remove its temporary input")
        let link = root.appendingPathComponent("linked.png")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        let linked = try PredictionImage(source: link, stamp: ImageFileStamp.read(link))
        defer { linked.remove() }
        try fixture(source, width: 6, height: 5)
        let regular = try FileManager.default.attributesOfItem(atPath: linked.url.path)[.type] as? FileAttributeType
        let linkedImage = try ImageCodec.decode(linked.url)
        check(regular == .typeRegular && linkedImage.width == 8, "Symbolic links must be dereferenced into independent worker input files")
    }

    private static func draftChecks(_ root: URL) async throws {
        let first = ImageSession(title: "Draft One", imageName: "one.png")
        let second = ImageSession(title: "Draft Two", imageName: "two.png")
        try LibraryStorage(directory: root).save(LibraryState(sessions: [first, second], selectedID: first.id))
        let store = StudioStore(directory: root)
        defer { store.shutdown() }
        store.question = "My unfinished draft"
        store.reuseQuestion("History question")
        let undo = store.notice!
        check(store.question == "History question" && undo.draftUndoID != nil, "Replacing an existing draft must offer undo")
        store.undoNotice(undo)
        check(store.question == "My unfinished draft", "Undo must recover the actual previous draft")
        store.reuseQuestion("History question")
        let late = store.notice!
        store.question = "New typing after reuse"
        store.undoNotice(late)
        check(store.question == "New typing after reuse", "Undo must never overwrite edits made after question reuse")
        store.question = "First image draft"
        store.reuseQuestion("Another history question")
        let otherImageUndo = store.notice!
        store.selectedID = second.id; store.question = "Second image draft"
        store.undoNotice(otherImageUndo)
        check(store.question == "Second image draft" && store.selectedID == second.id, "Undo from another image must not change its selection or draft")
        store.selectedID = first.id
        check(store.question == "First image draft", "Cross-image undo must restore only its original image's draft")
        store.question = ""; store.notice = nil
        store.reuseQuestion("Fill an empty draft")
        check(store.notice?.draftUndoID == nil && store.question == "Fill an empty draft", "An empty draft needs no destructive replacement notice")
        store.shutdown()
        let restored = StudioStore(directory: root)
        check(restored.question == "Fill an empty draft", "Question reuse must remain durable through normal quit")
        restored.shutdown()
    }

    private static func clipboardChecks(_ root: URL) async throws {
        let first = root.appendingPathComponent("first.png")
        let second = root.appendingPathComponent("second.png")
        try fixture(first, width: 4, height: 3)
        try fixture(second, width: 8, height: 7)
        let original = try Data(contentsOf: first)
        let tiff = NSBitmapImageRep(data: original)!.representation(using: .tiff, properties: [:])!
        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        let store = StudioStore(directory: root.appendingPathComponent("mixed-queue"))
        store.importImages([first])
        pasteboard.setData(original, forType: .png)
        store.pasteImage(from: pasteboard)
        pasteboard.clearContents(); pasteboard.setData(tiff, forType: .tiff)
        store.pasteImage(from: pasteboard)
        pasteboard.clearContents(); pasteboard.writeObjects([first as NSURL, second as NSURL])
        store.pasteImage(from: pasteboard)
        pasteboard.clearContents(); pasteboard.setString("clipboard changed after Paste", forType: .string)
        check(store.importing && store.sessions.isEmpty, "Paste conversion must join asynchronous import work instead of blocking the main actor")
        let quit = await store.prepareToTerminate()
        let saved = try LibraryStorage(directory: root.appendingPathComponent("mixed-queue")).load()
        check(quit && saved.sessions.count == 5, "Quit must drain accepted file, PNG, TIFF and Finder-file paste requests")
        check(saved.sessions.map(\.title) == ["second", "first", "剪贴板图片", "剪贴板图片", "first"], "Clipboard and file imports must retain one acceptance order")
        let dimensions = try saved.sessions.map { try ImageCodec.decode(store.imageURL(for: $0)).width }
        check(dimensions == [8, 4, 4, 4, 4], "Paste requests must retain their own captured bytes even after the clipboard changes")
        let after = try Data(contentsOf: first)
        check(after == original, "Pasting files must never alter their source bytes")
        store.pasteImage(from: pasteboard)
        check(!store.importing && store.sessions.count == 5, "Paste must not add work after quit has started")

        let partial = StudioStore(directory: root.appendingPathComponent("partial"))
        pasteboard.clearContents(); pasteboard.setData(Data("broken image fixture".utf8), forType: .png)
        partial.pasteImage(from: pasteboard)
        partial.importImages([second])
        let partialQuit = await partial.prepareToTerminate()
        check(partialQuit && partial.sessions.count == 1 && partial.selected?.title == "second", "A corrupt clipboard image must not discard later successful file imports")
        let files = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("partial/Images"), includingPropertiesForKeys: nil)
        check(files.count == 1 && partial.errorMessage?.contains("1 张") == true, "Failed clipboard requests must report failure without leaving owned orphan images")
        let empty = StudioStore(directory: root.appendingPathComponent("empty"))
        let emptyBoard = NSPasteboard.withUniqueName()
        defer { emptyBoard.releaseGlobally() }
        empty.pasteImage(from: emptyBoard)
        check(!empty.importing && empty.errorMessage?.contains("没有可用图片") == true, "An empty clipboard must explain why no import was accepted")
        empty.shutdown()
    }

    private static func importControlChecks(_ root: URL) async throws {
        let first = root.appendingPathComponent("first.png")
        let second = root.appendingPathComponent("second.png")
        let third = root.appendingPathComponent("third.png")
        for url in [first, second, third] { try fixture(url) }
        let entered = root.appendingPathComponent("entered-second")
        let release = ImportTestGate()
        let store = StudioStore(directory: root.appendingPathComponent("library"), importConverter: { request in
            if request.title == "second" {
                try Data().write(to: entered)
                await release.wait()
            }
            return try request.pngData()
        })
        store.ready = true
        store.importImages([first, second, third])
        try await waitForCondition { FileManager.default.fileExists(atPath: entered.path) }
        check(store.sessions.count == 1, "The first accepted image must be committed before controlling the next conversion")
        store.cancelImport(); store.cancelImport()
        check(store.stoppingImport && !store.canImport && store.ready, "Stopping imports must keep the model ready and block new import requests")
        store.importImages([first])
        await release.open()
        try await waitForCondition { !store.importing }
        check(store.sessions.map(\.title) == ["first"] && !store.stoppingImport && store.canImport, "Cancellation must retain completed images, skip the in-flight result and discard the queued tail")
        let images = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("library/Images"), includingPropertiesForKeys: nil)
        check(images.count == 1, "Cancelled conversion must not leave an unreferenced PNG")
        store.importImages([third])
        try await waitForCondition { !store.importing }
        check(store.sessions.map(\.title) == ["third", "first"], "A fresh import must work after cancellation without reviving old queued items")
        let quit = await store.prepareToTerminate()
        check(quit, "Normal quit must remain available after an import has been stopped")

        let drainingRoot = root.appendingPathComponent("quit-drain")
        let drainingEntered = root.appendingPathComponent("draining-second")
        let drainRelease = ImportTestGate()
        let draining = StudioStore(directory: drainingRoot, importConverter: { request in
            if request.title == "second" {
                try Data().write(to: drainingEntered)
                await drainRelease.wait()
            }
            return try request.pngData()
        })
        draining.importImages([first, second, third])
        try await waitForCondition { FileManager.default.fileExists(atPath: drainingEntered.path) }
        let quitting = Task { await draining.prepareToTerminate() }
        try await waitForCondition { draining.quitting }
        draining.cancelImport()
        check(!draining.stoppingImport, "A late Stop command must not discard accepted work after quit has promised to drain it")
        await drainRelease.open()
        let drained = await quitting.value
        let saved = try LibraryStorage(directory: drainingRoot).load()
        check(drained && saved.sessions.count == 3, "Quit must preserve the complete accepted queue when cancellation was not requested before quitting")
    }

    private static func importRollbackChecks(_ root: URL) async throws {
        let source = root.appendingPathComponent("source.png")
        try fixture(source)
        let folder = root.appendingPathComponent("library")
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("Images"), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: folder.appendingPathComponent("Images/owned.png"))
        let session = ImageSession(title: "Existing", imageName: "owned.png")
        try LibraryStorage(directory: folder).save(LibraryState(sessions: [session], favoriteIDs: [session.id], selectedID: session.id, favoritesOnly: true))
        let store = StudioStore(directory: folder)
        await store.preview.waitUntilLoaded()
        store.search = "Existing"; store.imageZoom = .double; store.question = "Keep this draft"
        store.retrySave()
        let snapshot = folder.appendingPathComponent("library.json")
        let original = try Data(contentsOf: snapshot)
        let marker = folder.appendingPathComponent(".library-established")
        let markerBytes = try Data(contentsOf: marker)
        try FileManager.default.removeItem(at: marker)
        try FileManager.default.createDirectory(at: marker, withIntermediateDirectories: false)
        store.importImages([source])
        try await waitForCondition { !store.importing }
        check(store.search == "Existing" && store.imageZoom == .double && store.favoritesOnly, "Failed imports must restore transient search/zoom as well as persisted filters")
        check(store.selectedID == session.id && store.question == "Keep this draft" && store.sessions.count == 1, "Failed imports must restore the original selection, library and draft")
        let unchanged = try Data(contentsOf: snapshot)
        let images = try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("Images"), includingPropertiesForKeys: nil)
        check(unchanged == original && images.count == 1, "A failed authority write must not commit a session whose image is rolled back")
        try FileManager.default.removeItem(at: marker)
        try markerBytes.write(to: marker)
        store.retrySave(); store.shutdown()
    }

    private static func quitErrorPresentationChecks(_ root: URL) async throws {
        let first = root.appendingPathComponent("first.png")
        let writeFailure = root.appendingPathComponent("write-failure.png")
        let readFailure = root.appendingPathComponent("read-failure.png")
        try fixture(first); try fixture(writeFailure); try fixture(readFailure)
        let writeGate = ImportTestGate(), failureGate = ImportTestGate()
        let writeEntered = root.appendingPathComponent("write-entered")
        let failureEntered = root.appendingPathComponent("failure-entered")
        let library = root.appendingPathComponent("Library")
        let store = StudioStore(directory: library, importConverter: { request in
            if request.title == "write-failure" {
                try Data().write(to: writeEntered)
                await writeGate.wait()
            }
            if request.title == "read-failure" {
                try Data().write(to: failureEntered)
                await failureGate.wait()
                throw NSError(domain: "CONTROLLED_READ_FAILURE", code: 32)
            }
            return try request.pngData()
        })
        store.importImages([first, writeFailure, readFailure])
        try await waitForCondition { FileManager.default.fileExists(atPath: writeEntered.path) }
        check(store.sessions.count == 1, "Quit-error testing must start with a committed image")
        store.question = "Retain this draft through competing errors"
        store.retrySave()
        let snapshot = library.appendingPathComponent("library.json")
        let saved = try Data(contentsOf: snapshot)
        try FileManager.default.removeItem(at: snapshot)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let quit = Task { await store.prepareToTerminate() }
        try await waitForCondition { store.quitting }
        await writeGate.open()
        try await waitForCondition { FileManager.default.fileExists(atPath: failureEntered.path) }
        check(store.hasUnsavedChanges && store.errorMessage != nil, "A real import-save failure must be pending while the last converter is held")
        check(store.presentedErrorMessage == nil, "An intermediate import/save error must not open an alert during quit draining")
        await failureGate.open()
        let savedOnQuit = await quit.value
        check(!savedOnQuit && !store.quitting, "A failed final save must cancel quit")
        check(store.presentedErrorMessage?.contains("已暂缓退出") == true, "The visible final error must explain that unsaved data prevented quit")
        check(store.sessions.count == 1 && store.question == "Retain this draft through competing errors", "Competing failures must retain the committed image and current draft")
        try FileManager.default.removeItem(at: snapshot); try saved.write(to: snapshot)
        store.retrySave()
        check(store.presentedErrorMessage == nil && !store.hasUnsavedChanges, "Successful retry must clear the final error and warning")
        let finalQuit = await store.prepareToTerminate()
        check(finalQuit, "Quit must work after recovering competing failures")

        let ordinary = StudioStore(directory: root.appendingPathComponent("Ordinary"), importConverter: { _ in
            throw NSError(domain: "CONTROLLED_ORDINARY_FAILURE", code: 32)
        })
        ordinary.importImage(readFailure)
        try await waitForCondition { !ordinary.importing }
        check(ordinary.presentedErrorMessage?.contains("未能导入") == true, "Ordinary import failures must remain visible outside quit")
        _ = await ordinary.prepareToTerminate()

        let drainGate = ImportTestGate()
        let drainEntered = root.appendingPathComponent("drain-entered")
        let draining = StudioStore(directory: root.appendingPathComponent("SuccessfulQuit"), importConverter: { request in
            if request.title == "read-failure" {
                try Data().write(to: drainEntered)
                await drainGate.wait()
                throw NSError(domain: "CONTROLLED_DRAIN_FAILURE", code: 32)
            }
            return try request.pngData()
        })
        draining.importImages([first, readFailure])
        try await waitForCondition { FileManager.default.fileExists(atPath: drainEntered.path) }
        draining.errorMessage = "Previously visible error"
        check(draining.presentedErrorMessage != nil, "The fixture must begin with a visible prior error")
        let drainQuit = Task { await draining.prepareToTerminate() }
        try await waitForCondition { draining.quitting }
        check(draining.presentedErrorMessage == nil, "Beginning quit must suspend a prior error presentation")
        await drainGate.open()
        let drained = await drainQuit.value
        check(drained && draining.sessions.count == 1 && draining.errorMessage != nil && draining.presentedErrorMessage == nil,
              "A successful final save must quit without a late import-error dialog and retain accepted images")
    }

    private static func existingErrorQuitChecks(_ root: URL) async throws {
        let source = root.appendingPathComponent("source.png")
        try fixture(source)
        let library = root.appendingPathComponent("Library")
        let store = StudioStore(directory: library)
        store.importImage(source)
        try await waitForCondition { !store.importing }
        store.question = "Keep the draft when quitting from an existing dialog"
        store.retrySave()
        check(!store.suspendErrorForTermination(), "Ordinary quit must not wait for an absent error dialog")
        store.errorMessage = "Repeated error"
        let first = store.presentedError!
        store.errorMessage = "Repeated error"
        let second = store.presentedError!
        check(first.id != second.id, "Identical error text must still identify distinct occurrences")
        check(!store.dismissError(first) && store.presentedError?.id == second.id, "A late callback for an older identical error must retain the newer event")
        check(store.dismissError(second) && store.errorMessage == nil, "The matching ordinary dialog must dismiss normally")
        store.errorMessage = "Existing error before quit"
        let prior = store.presentedError!
        check(store.suspendErrorForTermination(), "Quit must begin dismissing an already-presented error")
        check(store.dismissingErrorForTermination && store.presentedError == nil && store.errorEvent?.id == prior.id,
              "The old error must be hidden while its native sheet closes, without discarding the event")
        check(!store.dismissError(prior) && store.errorMessage != nil, "The closing old dialog must not clear pending state during the quit handoff")
        let snapshot = library.appendingPathComponent("library.json")
        let saved = try Data(contentsOf: snapshot)
        try FileManager.default.removeItem(at: snapshot)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let failed = await store.prepareToTerminate()
        let final = store.presentedError!
        check(!failed && !store.quitting && !store.dismissingErrorForTermination, "A failed final save must restore error presentation after native dismissal")
        check(final.message.contains("已暂缓退出") && final.id != prior.id, "The new dialog must describe the cancelled quit rather than the old error")
        check(!store.dismissError(prior) && store.presentedError?.id == final.id, "A delayed original dismissal must not erase the failed-quit reason")
        check(!store.dismissError(nil) && store.errorMessage != nil, "An unrelated empty binding callback must not dismiss an error")
        check(store.suspendErrorForTermination(), "The failed-quit dialog must support another quit attempt")
        let repeated = await store.prepareToTerminate()
        let latest = store.presentedError!
        check(!repeated && latest.message == final.message && latest.id != final.id,
              "A repeated failed quit must retain a new event even when the displayed reason is identical")
        check(!store.dismissError(final) && store.presentedError?.id == latest.id, "The previous failed-quit dialog must not clear the newest identical failure")
        try FileManager.default.removeItem(at: snapshot); try saved.write(to: snapshot)
        store.retrySave()
        let restored = try LibraryStorage(directory: library).load()
        check(store.presentedError == nil && !store.hasUnsavedChanges && restored.sessions.count == 1,
              "Retry must clear the current event and retain the committed image")
        check(restored.drafts[store.selectedID!.uuidString] == store.question,
              "The original draft must remain durably saved after recovering repeated failed quits")
        let succeeded = await store.prepareToTerminate()
        check(succeeded, "Normal quit must remain available after recovering an existing error dialog")
    }

    private static func errorPresentationChecks() {
        let presentation = StudioErrorPresentation()
        let first = StudioErrorEvent(message: "First error")
        let second = StudioErrorEvent(message: "Newer error")
        let repeated = StudioErrorEvent(message: "Newer error")
        presentation.update(first, hasVisibleSheet: false)
        check(presentation.error == first, "The first pending error must present normally")
        presentation.update(second, hasVisibleSheet: true)
        check(presentation.error == first, "An open dialog must keep its immutable event and dismissal binding")
        check(!presentation.dismiss(second) && presentation.error == first, "An unrelated dismissal must not close the displayed event")
        check(presentation.dismiss(first) && presentation.error == nil, "Acknowledging the visible event must start closing its sheet")
        presentation.update(second, hasVisibleSheet: true)
        check(presentation.error == nil, "A pending newer error must wait until the old native sheet ends")
        presentation.sheetDidEnd(latest: second)
        check(presentation.error == second, "The newer error must present after native dismissal")
        check(!presentation.dismiss(first) && presentation.error == second, "A late original binding callback must not dismiss the new dialog")
        presentation.update(repeated, hasVisibleSheet: true)
        check(presentation.error == second, "Identical text must retain separate displayed and pending event identities")
        check(presentation.dismiss(second), "The second displayed event must remain dismissible")
        presentation.sheetDidEnd(latest: repeated)
        check(presentation.error == repeated, "An identical newer occurrence must still appear after acknowledgement")
        presentation.update(nil, hasVisibleSheet: true)
        check(presentation.error == nil, "Quit or resolved-error suppression must hide the active dialog")
        presentation.update(first, hasVisibleSheet: true)
        check(presentation.error == nil, "A new failure must wait while a programmatically dismissed sheet closes")
        presentation.sheetDidEnd(latest: first)
        check(presentation.error == first, "A failed-quit error must appear after the suspended sheet ends")
        presentation.update(nil, hasVisibleSheet: false)
        presentation.update(second, hasVisibleSheet: false)
        check(presentation.error == second, "Clearing an error before any native sheet exists must not stall later errors")
        presentation.sheetDidEnd(latest: first)
        check(presentation.error == second, "An unrelated sheet end must not replace an active error")
    }

    private static func saveErrorIdentityChecks(_ root: URL) async throws {
        let source = root.appendingPathComponent("source.png")
        try fixture(source)
        let library = root.appendingPathComponent("Library")
        let store = StudioStore(directory: library)
        store.importImage(source)
        try await waitForCondition { !store.importing }
        store.question = "Preserve this draft while resolving only save errors"
        store.retrySave()
        let snapshot = library.appendingPathComponent("library.json")
        func failSaving() throws -> Data {
            let bytes = try Data(contentsOf: snapshot)
            try FileManager.default.removeItem(at: snapshot)
            try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
            store.retrySave()
            check(store.hasUnsavedChanges && store.errorEvent != nil, "The controlled save failure must expose a recoverable error")
            return bytes
        }
        func restoreSaving(_ bytes: Data) throws {
            try FileManager.default.removeItem(at: snapshot)
            try bytes.write(to: snapshot)
        }
        let first = try failSaving()
        store.modelError = true
        store.errorMessage = "A newer unrelated model failure"
        let modelFailure = store.errorEvent!
        try restoreSaving(first)
        store.retrySave()
        check(!store.hasUnsavedChanges && store.errorEvent == modelFailure,
              "Saving successfully must retain a newer unrelated model error")
        check(store.modelError, "A successful save must not claim that a separate model failure recovered")
        check(store.dismissError(modelFailure), "The retained model failure must still be dismissible")
        let second = try failSaving()
        let oldSaveError = store.errorEvent!
        store.errorMessage = oldSaveError.message
        let matchingText = store.errorEvent!
        try restoreSaving(second)
        store.showInspector.toggle()
        try await waitForCondition { !store.hasUnsavedChanges }
        check(!store.hasUnsavedChanges && store.errorEvent == matchingText,
              "Automatic saving must use error identity rather than clearing a newer occurrence with identical text")
        check(store.dismissError(matchingText), "The identical-text new occurrence must retain normal acknowledgement")
        let third = try failSaving()
        try restoreSaving(third)
        store.retrySave()
        check(!store.hasUnsavedChanges && store.errorEvent == nil, "Retry must still clear the actual resolved save failure")
        let saved = try LibraryStorage(directory: library).load()
        check(saved.sessions.count == 1 && saved.drafts[store.selectedID!.uuidString] == store.question,
              "Error recovery must retain the committed image and exact current draft")
        let terminated = await store.prepareToTerminate()
        check(terminated, "Normal termination must remain available after save-error recovery")
    }

    private static func waitForCondition(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while !condition() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(condition(), "The isolated workflow must reach the expected state before its deadline")
    }

    private static func feedbackChecks() async throws {
        let feedback = CopyFeedback()
        feedback.confirm(for: .milliseconds(220))
        check(feedback.visible, "Successful copy feedback must appear immediately")
        try await Task.sleep(for: .milliseconds(150))
        feedback.confirm(for: .milliseconds(220))
        try await Task.sleep(for: .milliseconds(110))
        check(feedback.visible, "A repeated copy must restart feedback instead of being cleared by the old timer")
        try await Task.sleep(for: .milliseconds(160))
        check(!feedback.visible, "Copy feedback must expire after the latest copy")
        feedback.confirm(for: .milliseconds(100)); feedback.reset()
        try await Task.sleep(for: .milliseconds(150))
        check(!feedback.visible, "A disappearing answer must cancel its feedback timer")
    }

    private static func conversationChecks() {
        let record = QuestionRecord(question: "fixture", answers: [], elapsed: 0, model: "fixture", checkpoint: "fixture", device: "cpu")
        let session = ImageSession(title: "History", imageName: "fixture.png", questions: [record])
        check(ConversationScrollTarget.latest(in: session, pending: nil) == .question(record.id), "Conversation opening must target the last actual question")
        check(ConversationScrollTarget.latest(in: session, pending: PendingQuestion(sessionID: session.id, question: "pending")) == .pending, "A submitted question on this image must take precedence over old history")
        check(ConversationScrollTarget.latest(in: session, pending: PendingQuestion(sessionID: UUID(), question: "other")) == .question(record.id), "Another image's pending task must not move this conversation")
        check(ConversationScrollTarget.latest(in: nil, pending: nil) == nil, "No selection must not produce a phantom scroll target")
        let empty = ImageSession(title: "Empty", imageName: "empty.png")
        check(ConversationScrollTarget.latest(in: empty, pending: nil) == nil, "An empty conversation must retain its guide")
    }
}
