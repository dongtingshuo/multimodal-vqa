import Foundation

struct QuestionDrafts: Equatable {
    private var values: [UUID?: String] = [:]

    init(saved: [String: String] = [:]) {
        for (key, value) in saved where !value.isEmpty {
            if key == "unselected" { values[nil] = value }
            else if let id = UUID(uuidString: key) { values[id] = value }
        }
    }

    var saved: [String: String] {
        Dictionary(uniqueKeysWithValues: values.map { ($0.key?.uuidString ?? "unselected", $0.value) })
    }

    subscript(sessionID: UUID?) -> String {
        get { values[sessionID, default: ""] }
        set { values[sessionID] = newValue.isEmpty ? nil : newValue }
    }

    mutating func restore(_ submitted: String, for sessionID: UUID) {
        if self[sessionID].isEmpty { self[sessionID] = submitted }
    }

    mutating func remove(_ sessionID: UUID) { values.removeValue(forKey: sessionID) }
}
