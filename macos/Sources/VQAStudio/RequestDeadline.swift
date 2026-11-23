import Foundation

/// Resolves the response/deadline race without allowing an old timer to kill a later request.
final class RequestDeadline: @unchecked Sendable {
    private let lock = NSLock()
    private var active = true
    private var timedOut = false

    func expire() -> Bool {
        lock.withLock {
            guard active else { return false }
            active = false
            timedOut = true
            return true
        }
    }

    func finish() -> Bool {
        lock.withLock { active = false; return timedOut }
    }
}
