import Foundation

enum ImageSnapshotError: LocalizedError {
    case changed
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case .changed: "图片文件已发生变化，已更新预览。请确认图片后再次提问，原问题已保留。"
        case .unavailable(let detail): "无法准备本次分析的图片，原问题已保留。\n\(detail)"
        }
    }
}
