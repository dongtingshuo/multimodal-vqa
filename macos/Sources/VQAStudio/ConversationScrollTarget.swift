import Foundation

enum ConversationScrollTarget: Hashable {
    case question(UUID)
    case pending

    static func latest(in session: ImageSession?, pending: PendingQuestion?) -> Self? {
        guard let session else { return nil }
        if pending?.sessionID == session.id { return .pending }
        return session.questions.last.map { .question($0.id) }
    }
}
