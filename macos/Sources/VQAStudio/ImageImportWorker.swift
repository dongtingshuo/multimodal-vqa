import Foundation
import Darwin

/// Invoked before SwiftUI initializes, so image I/O never creates a second workspace or model.
enum ImageImportWorker {
    static let argument = "--vqa-image-import-worker"

    static func runIfRequested() -> Bool {
        guard CommandLine.arguments.dropFirst().first == argument else { return false }
        guard CommandLine.arguments.count == 3 else { Darwin.exit(64) }
        Darwin.exit(run(in: URL(fileURLWithPath: CommandLine.arguments[2])))
    }

    static func run(in directory: URL) -> Int32 {
        do {
            let data = try Data(contentsOf: directory.appendingPathComponent("request.json"))
            guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: String],
                  let path = payload["path"], path.hasPrefix("/") else {
                throw NSError(domain: "VQAStudio.ImageImport", code: 18,
                              userInfo: [NSLocalizedDescriptionKey: "无法读取图片导入请求。"])
            }
            let source = URL(fileURLWithPath: path)
            if payload["mode"] == "read" {
                let stamp = try ImageFileStamp.read(source)
                let known = payload["knownStamp"].flatMap { try? JSONDecoder().decode(ImageFileStamp.self, from: Data($0.utf8)) }
                let unchanged = known == stamp
                if !unchanged {
                    let maximum = payload["maximumPixelSize"].flatMap(Int.init)
                    let png = try ImageCodec.pngData(ImageCodec.decode(source, maximumPixelSize: maximum))
                    guard try ImageFileStamp.read(source) == stamp else { throw ImageReadError.changed }
                    try png.write(to: directory.appendingPathComponent("image.png"), options: .atomic)
                }
                let receipt = ImageReadReceipt(stamp: stamp, unchanged: unchanged)
                try JSONEncoder().encode(receipt).write(to: directory.appendingPathComponent("receipt.json"), options: .atomic)
            } else if payload["mode"] == "snapshot" {
                guard let encoded = payload["knownStamp"],
                      let stamp = try? JSONDecoder().decode(ImageFileStamp.self, from: Data(encoded.utf8)) else {
                    throw ImageSnapshotError.unavailable("无法读取图片副本请求。")
                }
                _ = try PredictionImage(source: source, stamp: stamp, directory: directory.appendingPathComponent("Snapshot"))
            } else {
                let png = try ImageImport.pngData(at: source)
                try png.write(to: directory.appendingPathComponent("image.png"), options: .atomic)
            }
            return 0
        } catch {
            var response = ["error": error.localizedDescription]
            if error is ImageReadError { response["kind"] = "changed" }
            if let snapshotError = error as? ImageSnapshotError {
                switch snapshotError {
                case .changed: response["kind"] = "snapshot-changed"
                case .unavailable(let detail): response["error"] = detail
                }
            }
            if let data = try? JSONSerialization.data(withJSONObject: response) {
                try? data.write(to: directory.appendingPathComponent("error.json"), options: .atomic)
            }
            return 1
        }
    }
}
