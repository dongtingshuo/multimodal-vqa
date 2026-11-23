import SwiftUI

struct StudioToolbarMaterial: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.studioMaterialPreferences) private var preferences

    @ViewBuilder
    func body(content: Content) -> some View {
        if reduceTransparency || contrast == .increased || preferences.reduceTransparency || preferences.increasedContrast {
            content.toolbarBackground(Color(nsColor: .windowBackgroundColor), for: .windowToolbar)
                .toolbarBackground(.visible, for: .windowToolbar)
        } else {
            // Keep one continuous titlebar surface so sidebar edges stay aligned.
            content.toolbarBackground(.bar, for: .windowToolbar)
                .toolbarBackground(.visible, for: .windowToolbar)
        }
    }
}
