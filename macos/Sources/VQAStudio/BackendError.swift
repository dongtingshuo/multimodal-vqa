import Foundation

enum BackendError: LocalizedError {
    case connection(String)
    case rejected(String)

    var errorDescription: String? {
        switch self {
        case .connection(let message), .rejected(let message): message
        }
    }

    var needsReload: Bool {
        if case .connection = self { return true }
        return false
    }
}
