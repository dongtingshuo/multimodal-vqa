import SwiftUI

struct QuitProgressView: View {
    let store: StudioStore

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) {
                message.fixedSize()
                Spacer(minLength: 12)
                stopButton.fixedSize()
            }
            VStack(alignment: .leading, spacing: 10) {
                message
                stopButton
            }.frame(maxWidth: .infinity, alignment: .leading)
        }.padding(.horizontal, 24).padding(.vertical, 10).background(.regularMaterial)
    }

    private var message: some View {
        HStack(spacing: 12) {
            ProgressView().controlSize(.small)
            Text(store.stoppingImport ? "正在停止导入并保存…" : store.importing ? "正在完成图片导入，保存后退出…" : "正在保存并退出…")
                .font(.callout).fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var stopButton: some View {
        if store.importing {
            Button(store.stoppingImport ? "停止中…" : "停止剩余导入并退出", role: .destructive, action: store.stopImportsForQuit)
                .buttonStyle(.bordered).disabled(store.stoppingImport)
                .help("保留已导入的图片，停止当前读取和剩余队列后退出。原图片文件不会被删除。")
        }
    }
}
