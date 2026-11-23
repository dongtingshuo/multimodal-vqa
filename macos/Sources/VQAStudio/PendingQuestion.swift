import Foundation

struct PendingQuestion: Codable, Equatable, Identifiable, Sendable {
    let id: UUID
    let sessionID: UUID
    let question: String

    init(id: UUID = UUID(), sessionID: UUID, question: String) {
        self.id = id; self.sessionID = sessionID; self.question = question
    }
}
