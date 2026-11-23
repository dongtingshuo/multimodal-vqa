import SwiftUI

struct WorkspaceDetailView: View {
    let store: StudioStore
    @State private var preferredQuestionWidth: CGFloat = 370

    var body: some View {
        GeometryReader { geometry in
            let minimumQuestionWidth: CGFloat = 320
            let minimumCanvasWidth: CGFloat = 370
            let availableForQuestions = geometry.size.width - minimumCanvasWidth - WorkspaceResizeHandle.hitWidth
            let maximumQuestionWidth = max(minimumQuestionWidth, min(480, availableForQuestions))
            let questionWidth = min(preferredQuestionWidth, maximumQuestionWidth)

            HStack(spacing: 0) {
                ImageCanvasView(store: store)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .accessibilityElement(children: .contain)
                    .accessibilityLabel("图片预览")
                if store.showInspector {
                    WorkspaceResizeHandle(width: questionWidth) { proposedWidth in
                        let adjustedWidth = max(minimumQuestionWidth, min(maximumQuestionWidth, proposedWidth))
                        if adjustedWidth != questionWidth { preferredQuestionWidth = adjustedWidth }
                    }
                    QuestionPanelView(store: store)
                        .frame(width: questionWidth)
                        .frame(maxHeight: .infinity)
                        .accessibilityElement(children: .contain)
                        .accessibilityLabel("问答面板")
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }
}
