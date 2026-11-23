import Foundation

struct ImageReadReceipt: Codable, Sendable {
    let stamp: ImageFileStamp
    let unchanged: Bool
}
