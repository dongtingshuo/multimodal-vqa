import SwiftUI

struct LibraryThumbnailView: View {
    let url: URL
    @State private var image: NSImage?
    @State private var loadedURL: URL?
    @State private var refresh = 0

    var body: some View {
        ZStack {
            if loadedURL == url, let image { Image(nsImage: image).resizable().scaledToFill() }
            else {
                StudioTheme.canvas
                Image(systemName: "photo").foregroundStyle(StudioTheme.secondary)
            }
        }.frame(width: 42, height: 42).clipShape(.rect(cornerRadius: 7)).accessibilityHidden(true)
            .task(id: "\(url.absoluteString)|\(refresh)") {
                let result = await StudioImageCache.shared.thumbnail(at: url)
                guard !Task.isCancelled else { return }
                loadedURL = url; image = result
            }
            .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in refresh += 1 }
    }
}
