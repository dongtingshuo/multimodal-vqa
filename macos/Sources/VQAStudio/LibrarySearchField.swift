import AppKit
import SwiftUI

struct LibrarySearchField: NSViewRepresentable {
    @Binding var text: String
    var focusRequest = 0

    func makeNSView(context: Context) -> FocusableSearchField {
        let field = FocusableSearchField(string: text)
        field.isBezeled = true
        field.bezelStyle = .roundedBezel
        field.placeholderAttributedString = NSAttributedString(
            string: "搜索图片、问题或答案",
            attributes: [.foregroundColor: NSColor(red: 0.37, green: 0.41, blue: 0.47, alpha: 1)]
        )
        field.sendsSearchStringImmediately = true
        field.delegate = context.coordinator
        field.target = context.coordinator
        field.action = #selector(Coordinator.changed(_:))
        field.setAccessibilityLabel("搜索本地历史")
        return field
    }

    func updateNSView(_ field: FocusableSearchField, context: Context) {
        context.coordinator.parent = self
        if field.stringValue != text { field.stringValue = text }
        if context.coordinator.lastFocusRequest != focusRequest {
            context.coordinator.lastFocusRequest = focusRequest
            if focusRequest > 0 { field.requestFocus() }
            else { field.cancelPendingFocus() }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    @MainActor
    final class Coordinator: NSObject, NSSearchFieldDelegate {
        var parent: LibrarySearchField
        var lastFocusRequest = 0
        init(parent: LibrarySearchField) { self.parent = parent }

        @objc func changed(_ sender: NSSearchField) { parent.text = sender.stringValue }

        func controlTextDidChange(_ notification: Notification) {
            if let field = notification.object as? NSSearchField { changed(field) }
        }
    }
}
