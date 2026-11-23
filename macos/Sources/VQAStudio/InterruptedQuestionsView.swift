import SwiftUI

struct InterruptedQuestionsView: View {
    let store: StudioStore

    var body: some View {
        Menu {
            Text("选择问题恢复；当前草稿会保留在此列表中。")
            ForEach(store.recoverableQuestions) { submission in
                Button(submission.question) { store.recoverQuestion(submission.id) }
            }
        } label: {
            Label("待恢复的问题（\(store.recoverableQuestions.count)）", systemImage: "arrow.uturn.backward.circle")
                .font(.caption)
        }
        .menuStyle(.borderlessButton).fixedSize(horizontal: false, vertical: true)
        .disabled(!store.canModifyLibrary || store.busy || store.quitting)
        .accessibilityLabel("恢复未完成的问题，共 \(store.recoverableQuestions.count) 条")
    }
}
