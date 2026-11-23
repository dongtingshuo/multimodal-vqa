import Foundation
import Darwin

@MainActor
final class LibraryStorage {
    let directory: URL
    private var loaded = false
    private var currentBytes: Data?
    private var legacyBytes: Data?

    init(directory: URL) { self.directory = directory }

    func load() throws -> LibraryState {
        try readLibrary(requireCurrent: false, requireExisting: false)
    }

    private var establishedURL: URL { directory.appendingPathComponent(".library-established") }

    private func missingPrimary() -> NSError {
        NSError(domain: "VQAStudio", code: 16, userInfo: [NSLocalizedDescriptionKey:
            "资料库主文件 library.json 缺失，保存已暂停。不会自动回退到较旧的历史备份，请恢复主文件后重新读取。"])
    }

    private func readLibrary(requireCurrent: Bool, requireExisting: Bool) throws -> LibraryState {
        let current = directory.appendingPathComponent("library.json")
        if let bytes = try readIfPresent(current) {
            let state = try JSONDecoder().decode(LibraryState.self, from: bytes)
            guard state.version == 1 else {
                throw NSError(domain: "VQAStudio", code: 10, userInfo: [NSLocalizedDescriptionKey: "资料库版本不受支持，请使用创建它的新版 App。"])
            }
            try validate(state)
            currentBytes = bytes; legacyBytes = nil; loaded = true
            return state
        }
        if requireCurrent { throw missingPrimary() }
        if try readIfPresent(establishedURL) != nil { throw missingPrimary() }
        let legacy = directory.appendingPathComponent("history.json")
        var state = LibraryState()
        let previous = try readIfPresent(legacy)
        if requireExisting && previous == nil {
            throw NSError(domain: "VQAStudio", code: 13, userInfo: [NSLocalizedDescriptionKey: "尚未找到恢复后的资料库文件，请先恢复原文件再重新读取。"])
        }
        if let previous {
            state.sessions = try JSONDecoder().decode([ImageSession].self, from: previous)
            state.selectedID = state.sessions.first?.id
        }
        try validate(state)
        currentBytes = nil; legacyBytes = previous; loaded = true
        return state
    }

    func save(_ state: LibraryState) throws {
        try validate(state)
        if !loaded { _ = try load() }
        let current = directory.appendingPathComponent("library.json")
        guard try readIfPresent(current) == currentBytes else { throw LibraryConflictError() }
        if currentBytes == nil {
            guard try readIfPresent(directory.appendingPathComponent("history.json")) == legacyBytes else {
                throw LibraryConflictError()
            }
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(state)
        // Establish authority before committing: no throwing operation may follow the snapshot write.
        if try readIfPresent(establishedURL) == nil {
            try Data("VQA Studio library format 1\n".utf8).write(to: establishedURL, options: .atomic)
        }
        try bytes.write(to: current, options: .atomic)
        currentBytes = bytes; legacyBytes = nil
    }

    func saveRecoveryCopy(_ state: LibraryState) throws -> URL {
        try validate(state)
        let folder = directory.appendingPathComponent("Recovery")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("library-recovery-\(UUID().uuidString).json")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(state).write(to: url, options: .atomic)
        return url
    }

    private func readIfPresent(_ url: URL) throws -> Data? {
        do { return try Data(contentsOf: url) }
        catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            // A dangling link is an existing broken source, not a new empty library.
            var attributes = stat()
            let exists = url.withUnsafeFileSystemRepresentation { path in
                guard let path else { return false }
                return Darwin.lstat(path, &attributes) == 0
            }
            if exists { throw error }
            return nil
        }
    }

    func reload(requireCurrent: Bool) throws -> LibraryState {
        try readLibrary(requireCurrent: requireCurrent, requireExisting: true)
    }

    private func validate(_ state: LibraryState) throws {
        let sessions = state.sessions + state.archived.map(\.session)
        let unfinished = state.interruptedQuestions ?? []
        let sessionIDs = Set(sessions.map(\.id))
        guard state.version == 1,
              Set(unfinished.map(\.id)).count == unfinished.count,
              unfinished.allSatisfy({ sessionIDs.contains($0.sessionID) && !$0.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
              Set(sessions.map(\.id)).count == sessions.count,
              sessions.allSatisfy({ session in
                  !session.imageName.isEmpty && session.imageName != "." && session.imageName != ".."
                      && !session.imageName.contains("/") && !session.imageName.contains("\0")
                      && Set(session.questions.map(\.id)).count == session.questions.count
                      && session.questions.allSatisfy { record in
                          record.elapsed.isFinite && record.elapsed >= 0
                              && Set(record.answers.map(\.id)).count == record.answers.count
                              && record.answers.allSatisfy { $0.score.isFinite && (0...1).contains($0.score) }
                      }
              }) else {
            throw NSError(domain: "VQAStudio", code: 12, userInfo: [NSLocalizedDescriptionKey: "资料库包含重复记录、无效答案或无效图片路径，原文件已保留。"])
        }
    }
}
