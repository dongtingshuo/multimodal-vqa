import AppKit

@MainActor
final class FocusableSearchField: NSSearchField {
    private var pendingFocus = false

    func requestFocus() {
        pendingFocus = true
        applyPendingFocus()
    }

    func cancelPendingFocus() { pendingFocus = false }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        applyPendingFocus()
    }

    private func applyPendingFocus() {
        if pendingFocus, let window, window.makeFirstResponder(self) { pendingFocus = false }
    }
}
