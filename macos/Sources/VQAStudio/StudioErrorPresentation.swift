import Foundation
import Observation

@Observable @MainActor
final class StudioErrorPresentation {
    private(set) var error: StudioErrorEvent?
    private var waitingForDismissal = false

    func update(_ latest: StudioErrorEvent?, hasVisibleSheet: Bool) {
        if latest == nil {
            if error != nil { waitingForDismissal = hasVisibleSheet }
            error = nil
        } else if error == nil && !waitingForDismissal {
            error = latest
        }
    }

    @discardableResult func dismiss(_ shown: StudioErrorEvent?) -> Bool {
        guard let shown, error?.id == shown.id else { return false }
        error = nil
        waitingForDismissal = true
        return true
    }

    func sheetDidEnd(latest: StudioErrorEvent?) {
        guard waitingForDismissal else { return }
        waitingForDismissal = false
        error = latest
    }
}
