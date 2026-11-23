import SwiftUI

struct MissingImageView: View {
    let problem: String?
    let reread: () -> Void
    let chooseImage: () -> Void

    var body: some View {
        ContentUnavailableView {
            Label("这张图片暂时不可用", systemImage: "photo.badge.exclamationmark")
        } description: {
            Text("已有问答和草稿仍然保留。原图恢复到本地后可重新读取，也可以导入图片开始新的问答。")
        } actions: {
            Button("重新读取图片", action: reread).buttonStyle(.borderedProminent)
            Button("导入图片…", action: chooseImage)
        }
        .help(problem ?? "")
    }
}
