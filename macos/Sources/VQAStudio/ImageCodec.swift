import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImageCodec {
    static func decode(_ url: URL, maximumPixelSize: Int? = nil) throws -> DecodedImage {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        return try autoreleasepool {
            _ = try ImageFileStamp.read(url)
            return try decodeSource(CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary), maximumPixelSize: maximumPixelSize)
        }
    }

    static func decode(_ data: Data) throws -> DecodedImage {
        try autoreleasepool {
            try decodeSource(CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary))
        }
    }

    private static func decodeSource(_ source: CGImageSource?, maximumPixelSize: Int? = nil) throws -> DecodedImage {
        guard let source,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
              width.intValue > 0, height.intValue > 0,
              let bitmap = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceShouldCacheImmediately: true,
                kCGImageSourceThumbnailMaxPixelSize: min(maximumPixelSize ?? max(width.intValue, height.intValue), max(width.intValue, height.intValue)),
              ] as CFDictionary) else {
            throw NSError(domain: "VQAStudio", code: 3, userInfo: [NSLocalizedDescriptionKey: "无法解码图片，请选择完整的 PNG、JPEG 或 HEIC 等图片文件。"])
        }
        return DecodedImage(bitmap: bitmap)
    }

    static func pngData(_ image: DecodedImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw NSError(domain: "VQAStudio", code: 15, userInfo: [NSLocalizedDescriptionKey: "无法创建图片副本。"])
        }
        CGImageDestinationAddImage(destination, image.bitmap, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw NSError(domain: "VQAStudio", code: 15, userInfo: [NSLocalizedDescriptionKey: "无法保存图片副本。"])
        }
        return data as Data
    }
}
