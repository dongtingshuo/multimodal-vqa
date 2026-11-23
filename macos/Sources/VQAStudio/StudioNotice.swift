import Foundation

struct StudioNotice: Identifiable {
    let id = UUID()
    let message: String
    var restoreID: UUID?
    var draftUndoID: UUID?
}
