import Foundation

/// A private per-request copy keeps the worker's input equal to the submitted preview.
struct PredictionImage: Sendable {
    let url: URL
    private let directory: URL

    /// Synchronous copying is restricted to the image helper (and focused codec checks).
    init(source: URL, stamp: ImageFileStamp, directory: URL? = nil) throws {
        let directory = directory ?? FileManager.default.temporaryDirectory.appendingPathComponent("vqa-prediction-\(UUID().uuidString)")
        do {
            guard try ImageFileStamp.read(source) == stamp else { throw ImageSnapshotError.changed }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            let copy = directory.appendingPathComponent("input." + source.pathExtension)
            try FileManager.default.copyItem(at: source.resolvingSymlinksInPath(), to: copy)
            guard try ImageFileStamp.read(source) == stamp else { throw ImageSnapshotError.changed }
            self.directory = directory; url = copy
        } catch {
            try? FileManager.default.removeItem(at: directory)
            if let error = error as? ImageSnapshotError { throw error }
            throw ImageSnapshotError.unavailable(error.localizedDescription)
        }
    }

    init(preparedDirectory: URL, sourceExtension: String) {
        directory = preparedDirectory
        url = preparedDirectory.appendingPathComponent("Snapshot").appendingPathComponent("input." + sourceExtension)
    }

    static func prepare(source: URL, stamp: ImageFileStamp) async throws -> Self {
        try await ImageImportClient(timeout: .seconds(15)).snapshot(at: source, stamp: stamp)
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }
}
