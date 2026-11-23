import Foundation

enum LibrarySortOrder: String, Codable, CaseIterable, Identifiable, Sendable {
    case imported, recentQuestions, name
    var id: Self { self }
    var title: String {
        switch self {
        case .imported: "最近导入"
        case .recentQuestions: "最近提问"
        case .name: "图片名称"
        }
    }
}
