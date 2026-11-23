import AppKit

// Keep the approved master unchanged; only add macOS icon-canvas breathing room.
guard CommandLine.arguments.count == 3,
      let artwork = NSImage(contentsOfFile: CommandLine.arguments[2]) else {
    fatalError("Usage: make_icon.swift <output-folder> <selected-icon.png>")
}
let folder = URL(fileURLWithPath: CommandLine.arguments[1]).appendingPathComponent("AppIcon.iconset")
try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
// The master tile spans 240/256. Match the 206/256 opaque body measured in
// this Mac's Calculator/Notes/Preview ICNS assets instead of filling the Dock slot.
let artworkScale = 103.0 / 120.0
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
            isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
            fatalError("Unable to allocate icon bitmap.")
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        context.imageInterpolation = .high
        context.cgContext.clear(CGRect(x: 0, y: 0, width: pixels, height: pixels))
        let edge = Double(pixels) * artworkScale
        let inset = (Double(pixels) - edge) / 2
        artwork.draw(in: NSRect(x: inset, y: inset, width: edge, height: edge))
        NSGraphicsContext.restoreGraphicsState()
        let suffix = scale == 2 ? "@2x" : ""
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            fatalError("Unable to encode icon PNG.")
        }
        try png.write(to: folder.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
    }
}
