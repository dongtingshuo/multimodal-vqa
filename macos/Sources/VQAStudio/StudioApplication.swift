import AppKit

@objc(StudioApplication)
@MainActor
final class StudioApplication: NSApplication {
    var handleWorkspaceSheetTermination: (() -> Bool)?
    var handleErrorTermination: (() -> Bool)?

    override func terminate(_ sender: Any?) {
        if StudioFilePanel.shared.cancelForTermination({ [weak self] in self?.terminate(nil) }) { return }
        if handleErrorTermination?() == true { return }
        if handleWorkspaceSheetTermination?() == true { return }
        super.terminate(sender)
    }

    func completePreparedTermination() {
        super.terminate(nil)
    }
}
