import SwiftUI

struct RenameImageView: View {
    let session: ImageSession
    let save: (String) -> Bool
    @State private var title: String
    @FocusState private var focused: Bool
    @Environment(\.dismiss) private var dismiss

    init(session: ImageSession, save: @escaping (String) -> Bool) {
        self.session = session
        self.save = save
        _title = State(initialValue: session.title)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("重命名图片").font(.title3.weight(.semibold))
            TextField("图片名称", text: $title).textFieldStyle(.roundedBorder).focused($focused).onSubmit(commit)
            Text("只更改资料库中的显示名称，图片和已有问答会保留。")
                .font(.caption).foregroundStyle(StudioTheme.secondary)
            HStack {
                Spacer()
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("保存", action: commit).buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }.padding(24).frame(width: 400).onAppear { focused = true }
    }

    private func commit() {
        let value = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }
        if save(value) { dismiss() }
    }
}
