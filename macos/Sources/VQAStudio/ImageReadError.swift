import Foundation

enum ImageReadError: LocalizedError {
    case changed
    var errorDescription: String? { "图片在读取时发生变化，请重新读取。" }
}
