import SwiftUI

struct PathSettingRow: View {
    let title: String
    @Binding var path: String
    var directory = false
    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(title).frame(width: 72, alignment: .leading)
            TextField("", text: $path)
                .labelsHidden().textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.leading)
                .frame(minWidth: 0, maxWidth: .infinity)
                .layoutPriority(1)
                .accessibilityLabel(title)
                .help(path)
            Button("选择…") {
                let panel = NSOpenPanel(); panel.canChooseDirectories = directory; panel.canChooseFiles = !directory
                StudioFilePanel.shared.present(panel) { response in
                    if response == .OK, let url = panel.url { path = url.path }
                }
            }.fixedSize()
        }
    }
}
