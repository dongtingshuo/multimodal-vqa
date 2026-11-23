import Foundation

enum ImageImportRequest: Sendable {
    case file(URL)
    case clipboard(Data)

    var title: String {
        switch self {
        case .file(let url): url.deletingPathExtension().lastPathComponent
        case .clipboard: "剪贴板图片"
        }
    }

    var displayName: String {
        switch self {
        case .file(let url): url.lastPathComponent
        case .clipboard: "剪贴板图片"
        }
    }

    func pngData() throws -> Data {
        switch self {
        case .file(let url): try ImageImport.pngData(at: url)
        case .clipboard(let data): try ImageCodec.pngData(ImageCodec.decode(data))
        }
    }
}
