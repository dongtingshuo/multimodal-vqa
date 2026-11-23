import SwiftUI

struct LibraryRecoveryView: View {
    let detail: String
    let canReload: Bool
    let openFolder: () -> Void
    let reload: () -> Void

    var body: some View {
        HStack(spacing: 16) {
            Label {
                VStack(alignment: .leading, spacing: 4) {
                    Text("资料库需要恢复，保存已暂停。")
                    Text(detail).lineLimit(3).textSelection(.enabled)
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                    Text("恢复有效的资料库文件后，点击“重新读取”。")
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                }
            } icon: { Image(systemName: "exclamationmark.triangle") }
            .fixedSize(horizontal: false, vertical: true).help(detail)
            Spacer(minLength: 0)
            Button("打开数据文件夹", action: openFolder)
            Button("重新读取", action: reload).disabled(!canReload)
        }.padding(.horizontal, 24).padding(.vertical, 12)
            .background(.yellow.opacity(0.12)).accessibilityElement(children: .contain)
    }
}
