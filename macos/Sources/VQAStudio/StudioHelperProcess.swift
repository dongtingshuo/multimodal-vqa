import Foundation
import Darwin

/// Owns one isolated file operation. Cancellation before launch prevents launch too.
final class StudioHelperProcess: @unchecked Sendable {
    private let lock = NSLock()
    private let process: Process
    private var cancelled = false
    private var started = false
    private let cleanup: StudioHelperCleanup

    init(executable: URL, workerArgument: String, directory: URL) {
        let cleanup = StudioHelperCleanup(directory: directory)
        self.cleanup = cleanup
        let process = Process()
        process.executableURL = executable
        process.arguments = [workerArgument, directory.path]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { _ in cleanup.childExited() }
        self.process = process
    }

    func start() throws {
        try lock.withLock {
            guard !cancelled else { cleanup.childExited(); throw CancellationError() }
            do { try process.run(); started = true }
            catch { cleanup.childExited(); throw error }
        }
    }

    var isRunning: Bool { lock.withLock { process.isRunning } }
    var terminationStatus: Int32 { lock.withLock { process.terminationStatus } }

    func stop() {
        lock.withLock {
            cancelled = true
            // The helper only writes its private staging directory; no user data needs flushing.
            if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }
        }
    }

    func release() {
        stop()
        if lock.withLock({ !started }) { cleanup.childExited() }
        cleanup.release()
    }
}
