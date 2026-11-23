import SwiftUI

struct WorkspaceHeader<Accessory: View>: View {
    let title: String
    let subtitle: String
    @ViewBuilder let accessory: Accessory

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 16) {
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.title2.weight(.semibold))
                    .lineLimit(1).truncationMode(.middle)
                    .textSelection(.enabled)
                Text(subtitle)
                    .font(.subheadline).foregroundStyle(StudioTheme.secondary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            accessory
        }
        .padding(.horizontal, 24).padding(.vertical, 20)
        .frame(minHeight: 88, alignment: .leading)
    }
}
