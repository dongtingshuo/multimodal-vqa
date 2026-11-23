import AppKit

@MainActor final class CachedThumbnail {
    let stamp: ImageFileStamp
    let image: NSImage

    let decoded: DecodedImage

    init(stamp: ImageFileStamp, decoded: DecodedImage) {
        self.stamp = stamp; self.decoded = decoded
        image = NSImage(cgImage: decoded.bitmap, size: NSSize(width: decoded.width, height: decoded.height))
    }
}
