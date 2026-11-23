import Foundation

struct StudioErrorEvent: Identifiable, Equatable, Sendable {
    let id = UUID()
    let message: String
}
