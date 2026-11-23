import CoreGraphics

// CGImage is immutable; the value crosses the decoder actor and is rendered without mutable NSImage caching.
struct DecodedImage: @unchecked Sendable {
    let bitmap: CGImage
    var width: Int { bitmap.width }
    var height: Int { bitmap.height }
}
