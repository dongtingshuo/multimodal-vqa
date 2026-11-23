import SwiftUI

struct StudioErrorRecoveryView: View {
    let store: StudioStore
    let error: StudioErrorEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(error.message)
                .font(.callout)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 12) {
                if store.modelError {
                    Button("重试加载模型", action: retryModel)
                        .disabled(store.busy || store.quitting)
                }
                Button("关闭提示") { store.dismissError(error) }
                    .disabled(store.quitting)
            }
        }
        .accessibilityElement(children: .contain)
    }

    private func retryModel() {
        guard !store.busy, !store.quitting, store.dismissError(error) else { return }
        Task { await store.loadModel() }
    }
}
