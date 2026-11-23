import AppKit
import SwiftUI

struct WorkspaceResizeHandle: View {
    static let hitWidth: CGFloat = 9
    private static let keyStep: CGFloat = 16

    let width: CGFloat
    var label = "问答面板宽度"
    var edge: HorizontalEdge = .trailing
    let resize: (CGFloat) -> Void
    @Environment(\.layoutDirection) private var direction
    @FocusState private var focused: Bool
    @State private var nativeFocus = false

    private var resizeSign: CGFloat {
        let trailingSign: CGFloat = direction == .leftToRight ? 1 : -1
        return edge == .trailing ? trailingSign : -trailingSign
    }

    var body: some View {
        Rectangle()
            .fill(focused || nativeFocus ? StudioTheme.accent : Color(nsColor: .separatorColor))
            .frame(width: 1)
            .frame(maxHeight: .infinity)
            .frame(width: Self.hitWidth, alignment: edge == .leading ? .trailing : .center)
            .overlay {
                DragRegion(width: width, sign: resizeSign, resize: resize, focusChanged: { nativeFocus = $0 })
                    .accessibilityHidden(true)
            }
            .focusable(interactions: .activate)
            .focused($focused)
            .onKeyPress(keys: [.leftArrow, .rightArrow], phases: [.down, .repeat]) { press in
                guard press.modifiers.isEmpty else { return .ignored }
                let step = press.key == .leftArrow ? Self.keyStep : -Self.keyStep
                resize(width + resizeSign * step)
                return .handled
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(label)
            .accessibilityValue("\(Int(width.rounded())) 点")
            .accessibilityAdjustableAction { adjustment in
                switch adjustment {
                case .increment: resize(width + Self.keyStep)
                case .decrement: resize(width - Self.keyStep)
                @unknown default: break
                }
            }
    }

    private struct DragRegion: NSViewRepresentable {
        let width: CGFloat
        let sign: CGFloat
        let resize: (CGFloat) -> Void
        let focusChanged: (Bool) -> Void
        @Environment(\.isEnabled) private var enabled

        func makeNSView(context: Context) -> DragView { DragView() }

        func updateNSView(_ view: DragView, context: Context) {
            view.panelWidth = width
            view.sign = sign
            view.resize = resize
            view.focusChanged = focusChanged
            view.allowsResize = enabled
        }
    }

    private final class DragView: NSView {
        var panelWidth: CGFloat = 0
        var sign: CGFloat = 1
        var resize: ((CGFloat) -> Void)?
        var focusChanged: ((Bool) -> Void)?
        var allowsResize = true {
            didSet {
                guard oldValue != allowsResize else { return }
                if !allowsResize { start = nil }
                window?.invalidateCursorRects(for: self)
            }
        }
        private var start: (x: CGFloat, width: CGFloat)?
        private var acceptsKeyboardFocus = false

        override var mouseDownCanMoveWindow: Bool { false }
        override var acceptsFirstResponder: Bool { allowsResize && acceptsKeyboardFocus }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { allowsResize }

        override func becomeFirstResponder() -> Bool {
            guard super.becomeFirstResponder() else { return false }
            Task { focusChanged?(true) }
            return true
        }

        override func resignFirstResponder() -> Bool {
            guard super.resignFirstResponder() else { return false }
            acceptsKeyboardFocus = false
            Task { focusChanged?(false) }
            return true
        }

        override func resetCursorRects() {
            super.resetCursorRects()
            guard allowsResize else { return }
            if #available(macOS 15.0, *) {
                addCursorRect(bounds, cursor: .columnResize)
            } else {
                addCursorRect(bounds, cursor: .resizeLeftRight)
            }
        }

        override func mouseDown(with event: NSEvent) {
            guard allowsResize else { return }
            start = (event.locationInWindow.x, panelWidth)
            // Accept focus only on demand so the divider does not take the initial text focus.
            acceptsKeyboardFocus = true
            if window?.makeFirstResponder(self) != true { acceptsKeyboardFocus = false }
        }

        override func mouseDragged(with event: NSEvent) {
            guard let start else { return }
            // Window coordinates stay stable while the handle itself moves.
            resize?(start.width + sign * (start.x - event.locationInWindow.x))
        }

        override func mouseUp(with event: NSEvent) {
            start = nil
        }

        override func keyDown(with event: NSEvent) {
            guard allowsResize, event.modifierFlags.intersection([.shift, .control, .option, .command]).isEmpty else {
                super.keyDown(with: event)
                return
            }
            switch event.specialKey {
            case .leftArrow: resize?(panelWidth + sign * WorkspaceResizeHandle.keyStep)
            case .rightArrow: resize?(panelWidth - sign * WorkspaceResizeHandle.keyStep)
            default: super.keyDown(with: event)
            }
        }
    }
}
