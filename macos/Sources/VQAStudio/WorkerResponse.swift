import Foundation

struct WorkerResponse: Decodable, Sendable {
    var id: String?
    var ok: Bool
    var error: String?
    var answers: [Prediction]?
    var elapsed: Double?
    var model: String?
    var device: String?
    var checkpoint: String?
}
