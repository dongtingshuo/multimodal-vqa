import Foundation

struct ArchivedImage: Codable, Identifiable, Equatable, Sendable {
    var session: ImageSession
    var removedAt = Date()
    var id: UUID { session.id }
}
