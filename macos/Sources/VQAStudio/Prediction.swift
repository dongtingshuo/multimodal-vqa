import Foundation

struct Prediction: Codable, Identifiable, Equatable, Sendable {
    var answer: String
    var score: Double
    var id: String { answer }
}
