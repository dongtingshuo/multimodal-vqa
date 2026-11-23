import Foundation
import Darwin

struct ImageFileStamp: Codable, Equatable, Sendable {
    let size: UInt64
    let modified: Date
    let fileNumber: UInt64

    static func read(_ url: URL) throws -> Self {
        var attributes = stat()
        // FileManager fetches extended attributes too; cloud providers may block that read.
        let result = url.withUnsafeFileSystemRepresentation { path in
            guard let path else { return Int32(-1) }
            return Darwin.fstatat(AT_FDCWD, path, &attributes, 0)
        }
        guard result == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        guard attributes.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), attributes.st_size >= 0 else {
            throw NSError(domain: "VQAStudio", code: 14, userInfo: [NSLocalizedDescriptionKey: "这不是可读取的图片文件。"])
        }
        let modified = Date(timeIntervalSince1970: Double(attributes.st_mtimespec.tv_sec) + Double(attributes.st_mtimespec.tv_nsec) / 1_000_000_000)
        return Self(size: UInt64(attributes.st_size), modified: modified, fileNumber: UInt64(attributes.st_ino))
    }
}
