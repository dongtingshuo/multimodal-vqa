import Foundation
import Observation

@MainActor @Observable
final class CopyFeedback {
    private(set) var visible = false
    @ObservationIgnored private var task: Task<Void, Never>?

    func confirm(for duration: Duration = .seconds(2)) {
        task?.cancel()
        visible = true
        task = Task { [weak self] in
            do { try await Task.sleep(for: duration) }
            catch { return }
            self?.visible = false
        }
    }

    func reset() { task?.cancel(); task = nil; visible = false }
}
