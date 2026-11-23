import AppKit
import Darwin
import Foundation

@MainActor @main
enum ImageImportChecks {
    private static var checks = 0

    private static func check(_ condition: @autoclosure () -> Bool, _ message: String) {
        precondition(condition(), message)
        checks += 1
    }

    static func main() async throws {
        // Only this test executable inserts a real blocking FIFO read before production decoding.
        // The production launcher has no test environment switch or blocked-input special case.
        if CommandLine.arguments.dropFirst().first == ImageImportWorker.argument {
            try blockTestInputIfRequested()
            if ImageImportWorker.runIfRequested() { return }
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent("isolated-import")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 4, pixelsHigh: 3, bitsPerSample: 8,
                                     samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                     colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        bitmap.bitmapData!.initialize(repeating: 255, count: bitmap.bytesPerRow * bitmap.pixelsHigh)
        let png = bitmap.representation(using: .png, properties: [:])!
        let source = root.appendingPathComponent("accepted.png")
        try png.write(to: source)
        let staging = root.appendingPathComponent("Staging")
        let client = ImageImportClient(temporaryDirectory: staging, timeout: .seconds(2))
        let data = try await client.pngData(for: .file(source))
        let decoded = try ImageCodec.decode(data)
        check(decoded.width == 4 && decoded.height == 3, "The production helper must return full-sized readable PNG pixels")
        let clipboard = try await client.pngData(for: .clipboard(png))
        let clipboardImage = try ImageCodec.decode(clipboard)
        check(clipboardImage.width == 4, "Captured clipboard bytes must use the same production helper")
        try await waitForCleanup(staging)

        let invalid = root.appendingPathComponent("invalid.png")
        try Data("not an image".utf8).write(to: invalid)
        do { _ = try await client.pngData(for: .file(invalid)); preconditionFailure("Invalid image must fail") }
        catch { check(error.localizedDescription.contains("无法解码"), "Decode errors must reach the parent with a usable explanation") }
        try await waitForCleanup(staging)

        let blocked = try blockingInput("blocked-cancel", root: root, png: png)
        let operation = Task { try await client.pngData(for: .file(blocked)) }
        let pid = try await waitForChild(blocked)
        let stoppedAt = ContinuousClock.now
        operation.cancel()
        do { _ = try await operation.value; preconditionFailure("Cancelled import must not return a PNG") }
        catch is CancellationError { check(true, "Cancellation must stay distinct from decode failure") }
        check(stoppedAt.duration(to: .now) < .seconds(1), "Cancellation must return without unblocking the native input")
        try await waitForExit(pid)
        try await waitForCleanup(staging)

        let crashed = try blockingInput("blocked-crash", root: root, png: png)
        let interrupted = Task { try await client.pngData(for: .file(crashed)) }
        let crashPID = try await waitForChild(crashed)
        _ = Darwin.kill(crashPID, SIGKILL)
        do { _ = try await interrupted.value; preconditionFailure("An unexpectedly terminated decoder must fail") }
        catch { check(error.localizedDescription.contains("意外中断"), "A decoder crash must surface a recoverable error instead of ending the main App") }
        try await waitForExit(crashPID)
        try await waitForCleanup(staging)

        let timed = try blockingInput("blocked-timeout", root: root, png: png)
        let bounded = ImageImportClient(temporaryDirectory: staging, timeout: .milliseconds(800))
        let timeoutAt = ContinuousClock.now
        do { _ = try await bounded.pngData(for: .file(timed)); preconditionFailure("Blocked input must time out") }
        catch { check(error.localizedDescription.contains("超时"), "A timed-out import must explain that a local/downloaded copy can be retried") }
        check(timeoutAt.duration(to: .now) < .seconds(3), "The deadline must bound the parent even when the child ignores SIGTERM")
        let timeoutPID = try Int32(String(contentsOf: pidFile(timed), encoding: .utf8))!
        try await waitForExit(timeoutPID)
        try await waitForCleanup(staging)
        _ = try await client.pngData(for: .file(source))
        check(true, "A new import must work after timeout")
        try await waitForCleanup(staging)

        let earlyDirectory = staging.appendingPathComponent("vqa-image-import-early")
        try FileManager.default.createDirectory(at: earlyDirectory, withIntermediateDirectories: true)
        let early = StudioHelperProcess(executable: Bundle.main.executableURL!, workerArgument: ImageImportWorker.argument, directory: earlyDirectory)
        early.stop()
        do { try early.start(); preconditionFailure("A cancelled-before-launch helper must never start") }
        catch is CancellationError { check(!early.isRunning, "Cancellation before launch must prevent a decoder process") }
        early.release()
        check(!FileManager.default.fileExists(atPath: earlyDirectory.path), "Cancellation before launch must not leak staging files")
        let missing = ImageImportClient(executable: root.appendingPathComponent("missing-executable"), temporaryDirectory: staging)
        do { _ = try await missing.pngData(for: .file(source)); preconditionFailure("Missing helper must fail") }
        catch { check(true, "A launch error must reach the caller") }
        try await waitForCleanup(staging)

        try await retainedImageChecks(root: root, png: png, client: client)
        try await storeStopChecks(root: root, png: png, client: client)
        try await quitChecks(root: root, png: png, client: bounded)
        let unchanged = try Data(contentsOf: source)
        check(unchanged == png, "Helper operations must never change the source image")
        await ImageDecodingService.shared.shutdown(); await ImageDecodingService.thumbnails.shutdown()
        print("\(checks) isolated import-process checks passed (real blocked native FIFO read; no cloud outage or model inference simulated as real)")
    }

    private static func blockTestInputIfRequested() throws {
        let job = URL(fileURLWithPath: CommandLine.arguments[2])
        let payload = try JSONSerialization.jsonObject(with: Data(contentsOf: job.appendingPathComponent("request.json"))) as! [String: String]
        let source = URL(fileURLWithPath: payload["path"]!)
        if source.lastPathComponent == "changing-test.png" {
            let counter = source.appendingPathExtension("attempts")
            var attempts = (try? Data(contentsOf: counter)) ?? Data(); attempts.append(1)
            try attempts.write(to: counter)
            try JSONSerialization.data(withJSONObject: ["error": "图片在读取时发生变化，请重新读取。", "kind": "changed"])
                .write(to: job.appendingPathComponent("error.json"))
            Darwin.exit(1)
        }
        guard source.lastPathComponent.hasPrefix("blocked-") || FileManager.default.fileExists(atPath: source.appendingPathExtension("block").path) else { return }
        signal(SIGTERM, SIG_IGN)
        try Data(String(getpid()).utf8).write(to: pidFile(source))
        let descriptor = Darwin.open(source.appendingPathExtension("fifo").path, O_RDONLY)
        if descriptor >= 0 {
            var byte: UInt8 = 0
            _ = Darwin.read(descriptor, &byte, 1)
            _ = Darwin.close(descriptor)
        }
    }

    private static func blockingInput(_ name: String, root: URL, png: Data) throws -> URL {
        let source = root.appendingPathComponent(name + ".png")
        try png.write(to: source)
        precondition(mkfifo(source.appendingPathExtension("fifo").path, 0o600) == 0)
        return source
    }

    private static func pidFile(_ source: URL) -> URL { source.appendingPathExtension("pid") }

    private static func waitForChild(_ source: URL) async throws -> Int32 {
        let marker = pidFile(source)
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while !FileManager.default.fileExists(atPath: marker.path) && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        check(FileManager.default.fileExists(atPath: marker.path), "The test child must enter its real native read before Stop")
        return try Int32(String(contentsOf: marker, encoding: .utf8))!
    }

    private static func waitForExit(_ pid: Int32) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while Darwin.kill(pid, 0) == 0 && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(Darwin.kill(pid, 0) != 0 && errno == ESRCH, "The cancelled helper must be reaped without an orphan process")
    }

    private static func waitForCleanup(_ directory: URL) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while !(try FileManager.default.contentsOfDirectory(atPath: directory.path)).isEmpty && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        check(files.isEmpty, "Successful, failed and cancelled imports must release their private staging files")
    }

    private static func waitForImport(_ store: StudioStore) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while store.importing && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
        check(!store.importing, "The parent import queue must finish within its deadline")
    }

    private static func retainedImageChecks(root: URL, png: Data, client: ImageImportClient) async throws {
        let source = root.appendingPathComponent("accepted.png")
        let first = try await client.readImage(at: source)
        let sourceStamp = try ImageFileStamp.read(source)
        check(first.data != nil && first.receipt.stamp == sourceStamp && !first.receipt.unchanged, "Helper pixels must carry the identity of the exact source read")
        let unchanged = try await client.readImage(at: source, knownStamp: first.receipt.stamp)
        check(unchanged.data == nil && unchanged.receipt.unchanged, "An unchanged read must reuse cached pixels without generating another PNG")
        let small = try await client.readImage(at: source, maximumPixelSize: 2)
        let smallImage = try ImageCodec.decode(small.data!)
        check(max(smallImage.width, smallImage.height) <= 2, "Thumbnail helpers must downsample before sending pixels back to the UI")

        let previewSource = try blockingInput("blocked-preview", root: root, png: png)
        let previewService = ImageDecodingService(client: client)
        let preview = ImagePreview(service: previewService)
        preview.select(previewSource)
        let previewPID = try await waitForChild(previewSource)
        var heartbeats = 0
        let heartbeat = Task { @MainActor in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .milliseconds(10)) } catch { return }
                heartbeats += 1
            }
        }
        try await Task.sleep(for: .milliseconds(100)); heartbeat.cancel()
        check(heartbeats >= 3 && preview.loading && preview.image == nil, "A blocked native preview read must keep the main actor responsive")
        let switched = ContinuousClock.now
        preview.select(source)
        await preview.waitUntilLoaded()
        check(switched.duration(to: .now) < .seconds(1) && preview.image?.width == 4 && preview.url == source,
              "Switching away from a blocked image must cancel it and show the next preview without waiting for timeout")
        try await waitForExit(previewPID)
        let original = preview.image!.bitmap
        preview.select(source); await preview.waitUntilLoaded()
        check(preview.image!.bitmap === original && preview.stamp == sourceStamp, "Activation checks must preserve the exact unchanged preview bitmap")

        let changed = root.appendingPathComponent("changing-test.png")
        try png.write(to: changed)
        preview.select(changed); await preview.waitUntilLoaded()
        try await Task.sleep(for: .milliseconds(100))
        let attempts = try Data(contentsOf: changed.appendingPathExtension("attempts"))
        check(!preview.loading && preview.image == nil && preview.problem?.contains("读取时发生变化") == true && attempts.count == 1,
              "A continuously changing source must produce a retryable state rather than an unbounded automatic reread loop")
        preview.select(source); await preview.waitUntilLoaded()
        check(preview.image?.width == 4 && preview.problem == nil, "A failed retained-image read must not poison the next selection")

        let thumbnailService = ImageDecodingService(client: client, maximumConcurrent: 2, maximumPixelSize: 84)
        let cache = StudioImageCache(service: thumbnailService)
        let cachedURL = root.appendingPathComponent("cached-thumbnail.png")
        try png.write(to: cachedURL)
        let cached = await cache.thumbnail(at: cachedURL)
        check(cached != nil, "The cache must display a successfully decoded helper thumbnail")
        let blockFlag = cachedURL.appendingPathExtension("block")
        try Data().write(to: blockFlag)
        precondition(mkfifo(cachedURL.appendingPathExtension("fifo").path, 0o600) == 0)
        let pending = Task { await cache.thumbnail(at: cachedURL) }
        let thumbnailPID = try await waitForChild(cachedURL)
        pending.cancel(); let cancelled = await pending.value
        check(cancelled == nil, "A disappeared row must cancel its own thumbnail request")
        try await waitForExit(thumbnailPID)
        try FileManager.default.removeItem(at: blockFlag)
        let reused = await cache.thumbnail(at: cachedURL)
        check(reused === cached, "Cancelling a row refresh must not discard its valid shared cache entry")

        let manyClient = ImageImportClient(temporaryDirectory: client.temporaryDirectory, timeout: .seconds(5))
        let bounded = ImageDecodingService(client: manyClient, maximumConcurrent: 2, maximumPixelSize: 84)
        let urls = try ["blocked-limit-a", "blocked-limit-b", "blocked-limit-c", "blocked-limit-d"].map { try blockingInput($0, root: root, png: png) }
        let initial = urls.prefix(2).map { url in Task { try await bounded.decode(url) } }
        let firstPID = try await waitForChild(urls[0]); let secondPID = try await waitForChild(urls[1])
        let queued = urls.suffix(2).map { url in Task { try await bounded.decode(url) } }
        let running = initial + queued
        try await Task.sleep(for: .milliseconds(100))
        check(!FileManager.default.fileExists(atPath: pidFile(urls[2]).path) && !FileManager.default.fileExists(atPath: pidFile(urls[3]).path),
              "A list must never launch more than two thumbnail helpers at once")
        let cancelledAt = ContinuousClock.now
        running[2].cancel()
        do { _ = try await running[2].value; preconditionFailure("Queued cancellation must throw") }
        catch is CancellationError { check(cancelledAt.duration(to: .now) < .seconds(1), "Cancelling a queued row must not wait behind blocked decoders") }
        running[0].cancel(); _ = try? await running[0].value
        try await waitForExit(firstPID)
        let fourthPID = try await waitForChild(urls[3])
        check(!FileManager.default.fileExists(atPath: pidFile(urls[2]).path), "A cancelled queued thumbnail must never launch later")
        await bounded.shutdown()
        for task in running { _ = try? await task.value }
        try await waitForExit(secondPID); try await waitForExit(fourthPID)
        do { _ = try await bounded.decode(source); preconditionFailure("A shutting-down decoder must reject new jobs") }
        catch is CancellationError { check(true, "Application shutdown must block new decoding jobs after draining cancelled work") }
        try await waitForCleanup(client.temporaryDirectory)
        let waitSource = try blockingInput("blocked-cancel-wait", root: root, png: png)
        preview.select(waitSource)
        let waitPID = try await waitForChild(waitSource)
        let waitStart = ContinuousClock.now
        await preview.cancelAndWait()
        check(!preview.loading && waitStart.duration(to: .now) < .seconds(1), "Quit must await preview cancellation without waiting for its blocked native read")
        try await waitForExit(waitPID)
        try await waitForCleanup(client.temporaryDirectory)
    }

    private static func storeStopChecks(root: URL, png: Data, client: ImageImportClient) async throws {
        let blocked = try blockingInput("blocked-batch", root: root, png: png)
        let source = root.appendingPathComponent("accepted.png")
        let store = StudioStore(directory: root.appendingPathComponent("Library"), importConverter: { try await client.pngData(for: $0) })
        store.ready = true
        store.importImages([source, blocked, source])
        let pid = try await waitForChild(blocked)
        check(store.sessions.count == 1, "The first successful image must already be committed before the blocked import")
        let id = store.selectedID
        store.question = "Retain this draft"
        store.cancelImport()
        try await waitForImport(store)
        try await waitForExit(pid)
        check(store.sessions.count == 1 && store.selectedID == id && store.question == "Retain this draft", "Stop must retain the committed image and draft while discarding the current image and queued tail")
        check(store.ready && !store.busy && !store.stoppingImport && store.canImport && store.errorMessage == nil, "Image cancellation must not stop the model or leave an error/disabled import state")
        store.importImages([source])
        try await waitForImport(store)
        check(store.sessions.count == 2, "Import must restart immediately after stopping a blocked native read")
        store.shutdown()
        let saved = try LibraryStorage(directory: root.appendingPathComponent("Library")).load()
        check(saved.sessions.count == 2 && saved.drafts[id!.uuidString] == "Retain this draft", "Accepted images and drafts must remain durable across restart")
        let images = try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("Library/Images").path)
        check(images.count == 2, "Cancelled imports must not leave an owned library PNG")
    }

    private static func quitChecks(root: URL, png: Data, client: ImageImportClient) async throws {
        let blocked = try blockingInput("blocked-quit", root: root, png: png)
        let store = StudioStore(directory: root.appendingPathComponent("QuitLibrary"), importConverter: { try await client.pngData(for: $0) })
        store.importImages([blocked, root.appendingPathComponent("accepted.png")])
        let pid = try await waitForChild(blocked)
        let started = ContinuousClock.now
        let quit = await store.prepareToTerminate()
        check(quit && !store.importing && started.duration(to: .now) < .seconds(4), "Quit must finish an unresponsive decoder through its timeout and drain the remaining accepted image")
        try await waitForExit(pid)
        let saved = try LibraryStorage(directory: root.appendingPathComponent("QuitLibrary")).load()
        check(saved.sessions.count == 1 && !store.hasUnsavedChanges, "Quit must preserve the successful remainder of a partially timed-out batch")
    }
}
