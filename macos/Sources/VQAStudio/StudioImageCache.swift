import AppKit

@MainActor
final class StudioImageCache {
    static let shared = StudioImageCache()
    private let thumbnails = NSCache<NSURL, CachedThumbnail>()
    private let service: ImageDecodingService
    private var requests: [URL: UUID] = [:]

    init(service: ImageDecodingService = .thumbnails) {
        self.service = service
        thumbnails.totalCostLimit = 8 * 1_024 * 1_024
    }

    func thumbnail(at url: URL) async -> NSImage? {
        let request = UUID(); requests[url] = request
        defer { if requests[url] == request { requests[url] = nil } }
        let cached = thumbnails.object(forKey: url as NSURL)
        do {
            let result = try await service.decode(url, cached: cached?.decoded, stamp: cached?.stamp)
            guard !Task.isCancelled else { return nil }
            guard requests[url] == request else { return thumbnails.object(forKey: url as NSURL)?.image }
            if let cached, result.stamp == cached.stamp, result.image.bitmap === cached.decoded.bitmap { return cached.image }
            let thumbnail = CachedThumbnail(stamp: result.stamp, decoded: result.image)
            thumbnails.setObject(thumbnail, forKey: url as NSURL, cost: result.image.bitmap.bytesPerRow * result.image.height)
            return thumbnail.image
        } catch {
            if !Task.isCancelled, requests[url] == request { thumbnails.removeObject(forKey: url as NSURL) }
            return nil
        }
    }

    func remove(at url: URL) {
        requests[url] = nil
        thumbnails.removeObject(forKey: url as NSURL)
    }
}
