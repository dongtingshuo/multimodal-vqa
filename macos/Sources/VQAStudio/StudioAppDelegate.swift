import AppKit

@MainActor
final class StudioAppDelegate: NSObject, NSApplicationDelegate {
    var prepareToTerminate: (@MainActor () async -> Bool)?
    var finishSheetTermination: (@MainActor () -> Void)?
    var showWorkspace: (@MainActor () -> Void)?
    var importImages: (@MainActor ([URL]) -> Void)? {
        didSet {
            guard let importImages, !pendingImages.isEmpty else { return }
            let images = pendingImages; pendingImages.removeAll()
            importImages(images)
        }
    }
    private var pendingImages: [URL] = []
    private var terminationInProgress = false
    private var terminationPrepared = false
    static let packagedIcon: NSImage? = Bundle.main.url(forResource: "FocusQ", withExtension: "icns")
        .flatMap { NSImage(contentsOf: $0) }

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let icon = Self.packagedIcon else { return }
        NSApplication.shared.applicationIconImage = icon
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        // A visible Settings window must not prevent the workspace from reopening.
        showWorkspace?()
        return true
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func application(_ application: NSApplication, open urls: [URL]) {
        showWorkspace?()
        receiveImages(urls)
    }

    func receiveImages(_ urls: [URL]) {
        let images = urls.filter(\.isFileURL)
        guard !images.isEmpty else { return }
        if let importImages { importImages(images) }
        else { pendingImages.append(contentsOf: images) }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminationPrepared { return .terminateNow }
        guard !terminationInProgress else { return .terminateCancel }
        guard let prepareToTerminate else { return .terminateNow }
        terminationInProgress = true
        Task { @MainActor in
            let saved = await prepareForTermination(prepareToTerminate)
            terminationInProgress = false
            sender.reply(toApplicationShouldTerminate: saved)
        }
        return .terminateLater
    }

    func requestSheetTermination() {
        guard !terminationInProgress, let prepareToTerminate, let finishSheetTermination else { return }
        terminationInProgress = true
        Task { @MainActor in
            if await prepareForTermination(prepareToTerminate) { finishSheetTermination() }
            else { terminationInProgress = false }
        }
    }

    private func prepareForTermination(_ prepare: @MainActor () async -> Bool) async -> Bool {
        guard await prepare() else { return false }
        await ImageDecodingService.shared.shutdown()
        await ImageDecodingService.thumbnails.shutdown()
        terminationPrepared = true
        return true
    }
}
