import SwiftUI

struct QuestionGuideView: View {
    @Bindable var store: StudioStore
    let suggestions = ["What is in the image?", "What color is the object?", "How many people are there?"]
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Image(systemName: "text.bubble").font(.system(size: 30, weight: .light)).foregroundStyle(StudioTheme.secondary).accessibilityHidden(true)
            Text("每个细节，都可以问一问。").font(.headline)
            Text(store.selected == nil ? "导入一张图片，即可用英文提问。" : "用英文描述你想知道的内容。模型会返回相关的候选答案。")
                .font(.subheadline).foregroundStyle(StudioTheme.secondary).lineSpacing(5)
            Text("试着这样问").font(.caption.weight(.semibold)).foregroundStyle(StudioTheme.secondary).padding(.top, 12)
            ForEach(suggestions, id: \.self) { suggestion in
                Button { store.reuseQuestion(suggestion) } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(suggestion).font(.subheadline).multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "arrow.up.left").font(.caption).fixedSize()
                    }
                        .padding(.vertical, 10).padding(.horizontal, 12).background(StudioTheme.canvas, in: .rect(cornerRadius: 8))
                }.buttonStyle(.plain).disabled(store.selected == nil || !store.canModifyLibrary || store.quitting)
            }
        }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 14)
    }
}
