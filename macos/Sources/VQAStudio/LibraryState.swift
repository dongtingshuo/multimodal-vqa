import Foundation

struct LibraryState: Codable, Equatable, Sendable {
    var version = 1
    var sessions: [ImageSession] = []
    var archived: [ArchivedImage] = []
    var drafts: [String: String] = [:]
    var favoriteIDs: Set<UUID> = []
    var selectedID: UUID?
    var showInspector = true
    var sortOrder: LibrarySortOrder = .imported
    var favoritesOnly = false
    // Optional for backwards-compatible decoding of the existing version-1 library.
    var interruptedQuestions: [PendingQuestion]?
}
