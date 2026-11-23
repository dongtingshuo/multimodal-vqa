import SwiftUI

struct AnswerView: View {
    let record: QuestionRecord
    let onReuse: (String) -> Void
    @State private var copyFeedback = CopyFeedback()
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack(alignment: .firstTextBaseline) {
                Text(record.question).font(.headline).textSelection(.enabled)
                Spacer(minLength: 8)
                Button {
                    copy("\(record.question)\n" + record.answers.map { "\($0.answer): \(String(format: "%.4f", $0.score))" }.joined(separator: "\n"))
                } label: {
                    Label(copyFeedback.visible ? "已复制" : "复制答案", systemImage: copyFeedback.visible ? "checkmark" : "doc.on.doc")
                        .frame(width: 20, height: 20).contentShape(.rect)
                }.labelStyle(.iconOnly).buttonStyle(.plain)
                    .foregroundStyle(copyFeedback.visible ? StudioTheme.accent : StudioTheme.secondary)
                    .help(copyFeedback.visible ? "已复制到剪贴板" : "复制问题和候选答案")
                    .onDisappear(perform: copyFeedback.reset)
                Menu("问答操作", systemImage: "ellipsis") {
                    Button("再次提问") { onReuse(record.question) }
                    if let first = record.answers.first { Button("复制最佳答案") { copy(first.answer) } }
                }.labelStyle(.iconOnly).menuStyle(.borderlessButton).menuIndicator(.hidden)
                    .frame(width: 20, height: 20).help("再次提问或复制最佳答案")
            }
            if let first = record.answers.first {
                Text(first.answer).font(.system(size: 32, weight: .semibold)).foregroundStyle(StudioTheme.ink).textSelection(.enabled)
            }
            VStack(spacing: 13) {
                ForEach(Array(record.answers.enumerated()), id: \.element.id) { index, prediction in
                    VStack(spacing: 5) {
                        HStack(alignment: .firstTextBaseline, spacing: 12) {
                            Text(prediction.answer).font(.subheadline).lineLimit(2).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text(prediction.score, format: .percent.precision(.fractionLength(1)))
                                .font(.caption.monospacedDigit()).foregroundStyle(StudioTheme.secondary)
                                .frame(width: 48, alignment: .trailing)
                        }
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                Capsule().fill(StudioTheme.canvas)
                                Capsule().fill(index == 0 ? StudioTheme.accent : StudioTheme.accent.opacity(0.30))
                                    .frame(width: max(0, geo.size.width * min(1, max(0, prediction.score))))
                            }
                        }.frame(height: 4).accessibilityHidden(true)
                    }.accessibilityElement(children: .combine)
                }
            }
            HStack {
                Text(record.model.uppercased()).lineLimit(1)
                Text(record.date, style: .time).help(record.date.formatted(date: .complete, time: .shortened))
                Spacer()
                Text(record.elapsed, format: .number.precision(.fractionLength(2))) + Text(" 秒")
            }.font(.caption2).foregroundStyle(StudioTheme.secondary)
            Divider().padding(.top, 6)
        }
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        if NSPasteboard.general.setString(text, forType: .string) { copyFeedback.confirm() }
    }
}
