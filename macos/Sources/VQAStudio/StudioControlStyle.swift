import SwiftUI

struct StudioControlStyle: ViewModifier {
    var prominent = false
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.studioMaterialPreferences) private var preferences

    @ViewBuilder
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *), !reduceTransparency, contrast != .increased,
           !preferences.reduceTransparency, !preferences.increasedContrast {
            if prominent { content.buttonStyle(.glassProminent).tint(StudioTheme.accent) }
            else { content.buttonStyle(.glass) }
        } else {
            if prominent { content.buttonStyle(.borderedProminent).tint(StudioTheme.accent) }
            else { content.buttonStyle(.bordered) }
        }
    }
}
