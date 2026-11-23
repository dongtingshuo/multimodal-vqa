import Foundation

struct LibraryConflictError: LocalizedError {
    var errorDescription: String? {
        "资料库文件已在其他位置被修改或移动，已暂停保存以避免覆盖。当前修改仍保留在窗口中，可保存恢复副本后重新读取。"
    }
}
