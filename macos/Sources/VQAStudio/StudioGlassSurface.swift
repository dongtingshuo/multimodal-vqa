import SwiftUI

struct StudioGlassSurface: ViewModifier {
    var cornerRadius: CGFloat = 14
    var interactive = false
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast
    @Environment(\.studioMaterialPreferences) private var preferences

    @ViewBuilder
    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius)
        let increasedContrast = contrast == .increased || preferences.increasedContrast
        if reduceTransparency || preferences.reduceTransparency || increasedContrast {
            content
                .background(Color(nsColor: .controlBackgroundColor), in: shape)
                .overlay { shape.strokeBorder(Color(nsColor: .separatorColor), lineWidth: increasedContrast ? 1.5 : 0.5) }
        } else if #available(macOS 26.0, *) {
            content.glassEffect(.regular.interactive(interactive && !reduceMotion && !preferences.reduceMotion), in: shape)
        } else {
            content.background(.regularMaterial, in: shape)
                .overlay { shape.strokeBorder(.quaternary) }
        }
    }
}
