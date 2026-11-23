import SwiftUI

struct PendingAnswerView: View {
    let question: String

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(question).font(.headline).textSelection(.enabled)
            HStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("正在观察图片…").font(.subheadline).foregroundStyle(StudioTheme.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
    }
}
