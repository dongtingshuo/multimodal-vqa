import Foundation

struct DraftReplacement {
    let id = UUID()
    let sessionID: UUID?
    let previous: String
    let replacement: String
}
