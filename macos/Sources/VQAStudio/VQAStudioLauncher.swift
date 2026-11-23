import SwiftUI

@main @MainActor
enum VQAStudioLauncher {
    static func main() {
        SettingsValidationWorker.runIfRequested()
        if ImageImportWorker.runIfRequested() { return }
        _ = StudioApplication.shared
        VQAStudioApp.main()
    }
}
