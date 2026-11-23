import Foundation

struct QuestionRecord: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var question: String
    var answers: [Prediction]
    var elapsed: Double
    var date = Date()
    var model: String
    var checkpoint: String
    var device: String
}
