import Foundation

struct ImageSession: Codable, Identifiable, Equatable, Sendable {
    var id = UUID()
    var title: String
    var imageName: String
    var date = Date()
    var questions: [QuestionRecord] = []
}
