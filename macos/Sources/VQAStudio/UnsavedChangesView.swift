import SwiftUI

struct UnsavedChangesView: View {
    let conflict: Bool
    let canReload: Bool
    let retry: () -> Void
    let reload: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Label(conflict ? "资料库文件已在外部更改，当前修改仍保留在窗口中。" : "更改尚未保存，请检查磁盘空间或文件夹权限。", systemImage: "exclamationmark.triangle")
                .font(.callout).fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 12)
            if conflict { Button("保存副本并重新读取", action: reload).disabled(!canReload) }
            Button("重试保存", action: retry)
        }.padding(.horizontal, 24).padding(.vertical, 10)
            .background(.yellow.opacity(0.12))
            .accessibilityElement(children: .contain)
    }
}
