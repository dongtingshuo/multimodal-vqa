import Foundation

enum ImageImport {
    static func pngData(at url: URL) throws -> Data {
        try ImageCodec.pngData(ImageCodec.decode(url))
    }
}
