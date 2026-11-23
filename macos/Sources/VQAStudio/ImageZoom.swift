import Foundation

enum ImageZoom: Double, CaseIterable, Identifiable, Sendable {
    case fit = 0, quarter = 0.25, half = 0.5, actual = 1, double = 2, quadruple = 4
    var id: Self { self }
    var title: String { self == .fit ? "适合窗口" : "\(Int(rawValue * 100))%" }
    var larger: Self {
        if self == .fit { return .actual }
        return Self.allCases.first { $0.rawValue > rawValue } ?? self
    }
    var smaller: Self {
        if self == .fit { return .half }
        return Self.allCases.last { $0 != .fit && $0.rawValue < rawValue } ?? self
    }
}
