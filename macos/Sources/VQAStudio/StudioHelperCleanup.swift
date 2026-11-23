import Foundation

/// The parent releases outputs after consuming or abandoning them; a late child exit retries cleanup.
final class StudioHelperCleanup: @unchecked Sendable {
    private let lock = NSLock()
    private let directory: URL
    private var released = false

    init(directory: URL) { self.directory = directory }

    func childExited() {
        let remove = lock.withLock { released }
        if remove { try? FileManager.default.removeItem(at: directory) }
    }

    func release() {
        lock.withLock { released = true }
        // A cancelled child only writes inside this directory and never recreates its parent.
        // Do not depend on a later termination callback to clean up after the App exits.
        try? FileManager.default.removeItem(at: directory)
    }
}
