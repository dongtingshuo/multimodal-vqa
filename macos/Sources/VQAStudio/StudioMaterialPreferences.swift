import SwiftUI

// A view context may request stronger legibility; system accessibility preferences always win.
struct StudioMaterialPreferences: EnvironmentKey, Sendable {
    static let defaultValue = Self()
    var reduceTransparency = false
    var increasedContrast = false
    var reduceMotion = false
}

extension EnvironmentValues {
    var studioMaterialPreferences: StudioMaterialPreferences {
        get { self[StudioMaterialPreferences.self] }
        set { self[StudioMaterialPreferences.self] = newValue }
    }
}
