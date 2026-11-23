import Foundation

struct ImageImportClient: Sendable {
    let executable: URL
    let temporaryDirectory: URL
    let timeout: Duration

    init(executable: URL = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0]),
         temporaryDirectory: URL = FileManager.default.temporaryDirectory,
         timeout: Duration = .seconds(60)) {
        self.executable = executable
        self.temporaryDirectory = temporaryDirectory
        self.timeout = timeout
    }

    func pngData(for request: ImageImportRequest) async throws -> Data {
        let result = try await perform(request, options: [:])
        guard let data = result.data else { throw failure("无法读取图片副本，请重新导入。") }
        return data
    }

    func readImage(at url: URL, maximumPixelSize: Int? = nil, knownStamp: ImageFileStamp? = nil) async throws -> (data: Data?, receipt: ImageReadReceipt) {
        var options = ["mode": "read"]
        if let maximumPixelSize {
            guard (1...16_384).contains(maximumPixelSize) else { throw failure("图片预览尺寸无效。") }
            options["maximumPixelSize"] = String(maximumPixelSize)
        }
        if let knownStamp { options["knownStamp"] = String(decoding: try JSONEncoder().encode(knownStamp), as: UTF8.self) }
        let result = try await perform(.file(url), options: options)
        guard let receipt = result.receipt, !receipt.unchanged || receipt.stamp == knownStamp else {
            throw failure("无法核对图片状态，请重新读取。")
        }
        return (result.data, receipt)
    }

    func snapshot(at url: URL, stamp: ImageFileStamp) async throws -> PredictionImage {
        do {
            let options = ["mode": "snapshot", "knownStamp": String(decoding: try JSONEncoder().encode(stamp), as: UTF8.self)]
            let result = try await perform(.file(url), options: options)
            guard let snapshot = result.snapshot else { throw failure("图片副本未能完成，请重试。") }
            return snapshot
        } catch is CancellationError { throw CancellationError() }
        catch let error as ImageSnapshotError { throw error }
        catch { throw ImageSnapshotError.unavailable(error.localizedDescription) }
    }

    private func perform(_ request: ImageImportRequest, options: [String: String]) async throws -> (data: Data?, receipt: ImageReadReceipt?, snapshot: PredictionImage?) {
        try Task.checkCancellation()
        let directory = temporaryDirectory.appendingPathComponent("vqa-image-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let source: URL
            switch request {
            case .file(let url): source = url
            case .clipboard(let data):
                source = directory.appendingPathComponent("clipboard")
                try data.write(to: source, options: .atomic)
            }
            var values = options; values["path"] = source.path
            let payload = try JSONSerialization.data(withJSONObject: values)
            try payload.write(to: directory.appendingPathComponent("request.json"), options: .atomic)
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
        let process = StudioHelperProcess(executable: executable, workerArgument: ImageImportWorker.argument, directory: directory)
        var transferred = false
        defer { if !transferred { process.release() } }
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try process.start()
            let deadline = ContinuousClock.now.advanced(by: timeout)
            while process.isRunning {
                if ContinuousClock.now >= deadline {
                    throw failure("图片读取超时。请确认图片已下载到本机后重试，或选择其他图片。")
                }
                try await Task.sleep(for: .milliseconds(20))
            }
            try Task.checkCancellation()
            guard process.terminationStatus == 0 else {
                let errorURL = directory.appendingPathComponent("error.json")
                if let data = try? Data(contentsOf: errorURL),
                   let result = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                   let message = result["error"] {
                    if result["kind"] == "changed" { throw ImageReadError.changed }
                    if result["kind"] == "snapshot-changed" { throw ImageSnapshotError.changed }
                    throw failure(message)
                }
                throw failure("图片读取意外中断，请重新导入这张图片。")
            }
            if options["mode"] == "snapshot", case .file(let source) = request {
                let snapshot = PredictionImage(preparedDirectory: directory, sourceExtension: source.pathExtension)
                guard let encoded = options["knownStamp"],
                      let stamp = try? JSONDecoder().decode(ImageFileStamp.self, from: Data(encoded.utf8)),
                      try ImageFileStamp.read(snapshot.url).size == stamp.size else {
                    throw failure("无法核对图片副本，请重新准备。")
                }
                try Task.checkCancellation()
                transferred = true
                return (nil, nil, snapshot)
            }
            let receipt = options["mode"] == "read"
                ? try JSONDecoder().decode(ImageReadReceipt.self, from: Data(contentsOf: directory.appendingPathComponent("receipt.json"))) : nil
            if receipt?.unchanged == true { return (nil, receipt, nil) }
            let data = try Data(contentsOf: directory.appendingPathComponent("image.png"))
            guard data.starts(with: [137, 80, 78, 71, 13, 10, 26, 10]) else {
                throw failure("无法读取图片副本，请重新导入。")
            }
            return (data, receipt, nil)
        } onCancel: { process.stop() }
    }

    private func failure(_ message: String) -> NSError {
        NSError(domain: "VQAStudio.ImageImport", code: 17, userInfo: [NSLocalizedDescriptionKey: message])
    }
}
