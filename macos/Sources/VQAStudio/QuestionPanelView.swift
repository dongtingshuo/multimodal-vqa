import SwiftUI

struct QuestionPanelView: View {
    @Bindable var store: StudioStore
    @FocusState private var focused: Bool
    @State private var scrollTarget: ConversationScrollTarget?

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader(title: "关于这张图片", subtitle: "提问、观察，再发现。") {
                Button("最新问答", systemImage: "arrow.down.to.line", action: store.showLatestQuestion)
                    .labelStyle(.iconOnly).buttonStyle(.plain).font(.title3).foregroundStyle(StudioTheme.accent)
                    .frame(width: 24, height: 24).help("跳到最新问答 ⌥⌘↓")
                    .disabled(store.selected?.questions.isEmpty != false && !store.isAnsweringSelectedImage)
            }
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        if store.selected?.questions.isEmpty != false && !store.isAnsweringSelectedImage { QuestionGuideView(store: store) }
                        ForEach(store.selected?.questions ?? []) { record in
                            AnswerView(record: record, onReuse: store.reuseQuestion).id(ConversationScrollTarget.question(record.id))
                        }
                        if store.isAnsweringSelectedImage, let pending = store.pendingQuestion {
                            PendingAnswerView(question: pending.question).id(ConversationScrollTarget.pending)
                        }
                    }.scrollTargetLayout().frame(maxWidth: .infinity, alignment: .leading).padding(24)
                }
                .defaultScrollAnchor(store.selected?.questions.isEmpty == false ? .bottom : .top)
                .scrollPosition(id: $scrollTarget, anchor: .bottom)
                .onChange(of: store.selectedID, initial: true) {
                    scrollTarget = ConversationScrollTarget.latest(in: store.selected, pending: store.pendingQuestion)
                }
                .onChange(of: store.selected?.questions.count) {
                    scrollTarget = ConversationScrollTarget.latest(in: store.selected, pending: store.pendingQuestion)
                }
                .onChange(of: store.pendingQuestion?.sessionID) {
                    if store.isAnsweringSelectedImage { scrollTarget = .pending }
                    else if scrollTarget == .pending {
                        scrollTarget = ConversationScrollTarget.latest(in: store.selected, pending: store.pendingQuestion)
                    }
                }
                .onChange(of: store.latestQuestionRequest) {
                    if let target = ConversationScrollTarget.latest(in: store.selected, pending: store.pendingQuestion) {
                        proxy.scrollTo(target, anchor: .bottom)
                    }
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                if !store.recoverableQuestions.isEmpty { InterruptedQuestionsView(store: store) }
                HStack {
                    Text("输入英文问题").font(.subheadline.weight(.medium))
                    Spacer()
                    Text("⌘ ↩").font(.caption).foregroundStyle(StudioTheme.secondary)
                }
                TextField("", text: $store.question, axis: .vertical)
                    .font(.body)
                    .textFieldStyle(.plain).lineLimit(3...6).focused($focused)
                    .overlay(alignment: Alignment(horizontal: .leading, vertical: .firstTextBaseline)) {
                        if store.question.isEmpty {
                            Text("What is in the image?").font(.body).foregroundStyle(StudioTheme.secondary)
                                .allowsHitTesting(false).accessibilityHidden(true)
                        }
                    }
                    .padding(13).background(StudioTheme.canvas, in: .rect(cornerRadius: 9))
                    .accessibilityLabel("输入关于当前图片的英文问题")
                    .disabled(!store.canModifyLibrary)
                    .task(id: store.questionFocusRequest) {
                        guard store.questionFocusRequest > 0, store.requestedFocus == .question else { return }
                        focused = false
                        await Task.yield()
                        guard !Task.isCancelled, store.requestedFocus == .question else { return }
                        focused = true
                    }
                HStack {
                    if store.busy && !store.ready {
                        ProgressView().controlSize(.small)
                        Text("模型加载中").font(.caption).foregroundStyle(StudioTheme.secondary)
                    } else {
                        Text(store.questionStatus)
                            .font(.caption).foregroundStyle(StudioTheme.secondary)
                    }
                    Spacer()
                    if store.busy {
                        Button(store.cancelling ? "停止中" : "停止", systemImage: "stop.fill", action: store.cancelModelTask)
                            .buttonStyle(.bordered).controlSize(.large).disabled(store.cancelling)
                            .help("停止当前任务，未完成的问题会保留。")
                    } else if !store.ready {
                        Button("加载模型") { Task { await store.loadModel() } }.modifier(StudioControlStyle(prominent: true)).controlSize(.large)
                    } else {
                        Button { Task { await store.ask() } } label: { Label("提问", systemImage: "arrow.up") }
                            .modifier(StudioControlStyle(prominent: true)).controlSize(.large).disabled(!store.canAsk)
                    }
                }
                Text("候选答案受模型词表限制，分数不代表正确率。")
                    .font(.caption).foregroundStyle(StudioTheme.secondary).fixedSize(horizontal: false, vertical: true)
            }.padding(.horizontal, 24).padding(.vertical, 20)
        }.background(.white)
    }
}
