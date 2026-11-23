import AppKit

@MainActor
final class StudioFilePanel {
    static let shared = StudioFilePanel()
    private var active: NSSavePanel?
    private var resumeTermination: (() -> Void)?

    func present(_ panel: NSSavePanel, completion: @escaping (NSApplication.ModalResponse) -> Void) {
        guard active == nil else {
            active?.makeKeyAndOrderFront(nil)
            return
        }
        active = panel
        let finished: (NSApplication.ModalResponse) -> Void = { [weak self] response in
            guard let self, self.active === panel else { return }
            self.active = nil
            let resume = self.resumeTermination
            self.resumeTermination = nil
            panel.orderOut(nil)
            completion(resume == nil ? response : .cancel)
            resume?()
        }
        if let owner = NSApplication.shared.keyWindow ?? NSApplication.shared.mainWindow {
            panel.beginSheetModal(for: owner, completionHandler: finished)
        } else {
            panel.begin(completionHandler: finished)
        }
    }

    func cancelForTermination(_ resume: @escaping () -> Void) -> Bool {
        guard let active else { return false }
        if resumeTermination == nil {
            resumeTermination = resume
            active.cancel(nil)
        }
        return true
    }
}
