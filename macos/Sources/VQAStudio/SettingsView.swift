import SwiftUI
import UniformTypeIdentifiers

struct SettingsView: View {
    let store: StudioStore
    @State private var draft: AppSettings
    @State private var validationIssue: String?

    init(store: StudioStore) {
        self.store = store
        _draft = State(initialValue: store.settings)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Image(systemName: "slider.horizontal.3").font(.largeTitle).foregroundStyle(StudioTheme.accent).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 5) {
                    Text("模型与运行环境").font(.title2.weight(.semibold))
                    Text("连接项目已有的 Python 环境和模型文件。 ").foregroundStyle(StudioTheme.secondary)
                }
                Spacer()
            }.padding(24)
            if store.quitting { QuitProgressView(store: store) }
            if store.hasUnsavedChanges {
                UnsavedChangesView(conflict: store.hasLibraryConflict, canReload: store.canReloadLibrary,
                                   retry: store.retrySave, reload: store.reloadLibrary)
                    .disabled(store.quitting)
            }
            Form {
                if let error = store.presentedError {
                    Section("需要处理") { StudioErrorRecoveryView(store: store, error: error) }
                }
                Section("本地文件") {
                    PathSettingRow(title: "项目目录", path: $draft.project, directory: true)
                    PathSettingRow(title: "Python", path: $draft.python)
                    PathSettingRow(title: "模型权重", path: $draft.checkpoint)
                    PathSettingRow(title: "配置文件", path: $draft.config)
                    if let validationIssue {
                        Label(validationIssue, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .disabled(store.checkingModelSettings || store.quitting)
                Section("推理") {
                    Picker("计算设备", selection: $draft.device) {
                        Text("CPU · 兼容优先").tag("cpu")
                        Text("Apple GPU · MPS").tag("mps")
                        Text("自动选择").tag("auto")
                    }
                    Stepper("候选答案：\(draft.topk) 个", value: $draft.topk, in: 1...10)
                    Toggle("仅使用本地缓存的模型资源", isOn: $draft.offline)
                    Text("关闭后，Python 模型库可能联网下载所需资源。MPS 支持取决于模型与 PyTorch 环境。")
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                }
                .disabled(store.checkingModelSettings || store.quitting)
                Section("应用数据") {
                    LabeledContent("草稿与历史", value: "自动保存在当前 Mac")
                    Button("打开本地数据文件夹", action: store.openDataFolder)
                    Button("打开运行日志", action: store.openLog)
                }
            }.formStyle(.grouped)
            HStack {
                if store.checkingModelSettings {
                    ProgressView("正在检查模型文件…").controlSize(.small)
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                } else {
                    Text(store.busy ? "请等待当前模型任务结束" : store.status)
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                }
                Spacer()
                if store.checkingModelSettings {
                    Button("停止检查", action: store.cancelModelTask)
                        .disabled(store.cancelling || store.quitting)
                }
                Button(applyTitle, action: applyDraft)
                    .buttonStyle(.borderedProminent).disabled(store.busy || store.quitting)
            }.padding(.horizontal, 24).padding(.vertical, 20)
        }.frame(width: 720, height: 760)
            .onAppear {
                draft = store.settings
                validationIssue = nil
            }
            .onChange(of: draft) { validationIssue = nil }
    }

    private var applyTitle: String {
        if draft.normalized == store.settings { return "重新加载模型" }
        return store.needsModelReload(for: draft) ? "应用并加载模型" : "应用设置"
    }

    private func applyDraft() {
        let submitted = draft
        validationIssue = nil
        Task {
            let issue = await store.apply(submitted)
            if draft == submitted {
                // Runtime failures already belong to the shared recovery section.
                validationIssue = issue == store.errorMessage ? nil : issue
            }
        }
    }
}
