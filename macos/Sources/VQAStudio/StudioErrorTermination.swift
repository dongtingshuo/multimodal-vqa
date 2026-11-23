import AppKit

@MainActor
final class StudioErrorTermination {
    static let shared = StudioErrorTermination()
    static let title = "需要处理"
    private var observer: NSObjectProtocol?
    private var resume: (() -> Void)?

    var hasVisibleErrorSheet: Bool { visibleErrorSheet != nil }

    private var visibleErrorSheet: NSWindow? {
        NSApplication.shared.windows.first {
            $0.isVisible && $0.sheetParent != nil && containsErrorTitle($0.contentView)
        }
    }

    func request(dismiss: () -> Bool, resume: @escaping () -> Void) -> Bool {
        if self.resume != nil { return true }
        // A pending error must not make us wait for an unrelated rename or archive sheet.
        if let sheet = visibleErrorSheet, let parent = sheet.sheetParent {
            self.resume = resume
            observer = NotificationCenter.default.addObserver(forName: NSWindow.didEndSheetNotification, object: parent, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.finish() }
            }
            if dismiss() { return true }
            clear()
            return false
        }
        guard dismiss() else { return false }
        resume()
        return true
    }

    private func containsErrorTitle(_ view: NSView?) -> Bool {
        guard let view else { return false }
        if let field = view as? NSTextField, !field.isEditable, field.stringValue == Self.title { return true }
        if let text = view as? NSTextView, !text.isEditable, text.string == Self.title { return true }
        return view.subviews.contains { containsErrorTitle($0) }
    }

    private func finish() {
        let completion = resume
        clear()
        completion?()
    }

    private func clear() {
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        resume = nil
    }
}
