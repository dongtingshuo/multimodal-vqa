import SwiftUI

struct WorkspaceView: View {
    @Bindable var store: StudioStore
    var sheetDidDismiss: () -> Void = {}
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var sidebarWidth: CGFloat = 235
    @State private var errorPresentation = StudioErrorPresentation()

    var body: some View {
        let error = errorPresentation.error
        return VStack(spacing: 0) {
            if store.quitting { QuitProgressView(store: store) }
            VStack(spacing: 0) {
                if let problem = store.libraryProblem {
                    LibraryRecoveryView(detail: problem, canReload: store.canReloadLibrary, openFolder: store.openDataFolder, reload: store.reloadLibrary)
                }
                if store.hasUnsavedChanges {
                    UnsavedChangesView(conflict: store.hasLibraryConflict, canReload: store.canReloadLibrary,
                                       retry: store.retrySave, reload: store.reloadLibrary)
                }
                workspace
            }.disabled(store.quitting)
        }
        .frame(minWidth: 1000, minHeight: 680)
        .onChange(of: store.searchFocusRequest) { columnVisibility = .all }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in store.refreshImagePreview() }
        .onChange(of: store.presentedError, initial: true) {
            errorPresentation.update(store.presentedError, hasVisibleSheet: StudioErrorTermination.shared.hasVisibleErrorSheet)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didEndSheetNotification)) { _ in
            errorPresentation.sheetDidEnd(latest: store.presentedError)
        }
        .sheet(item: $store.renameTarget, onDismiss: sheetDidDismiss) { session in
            VStack(spacing: 0) {
                if store.quitting { QuitProgressView(store: store) }
                if store.hasUnsavedChanges {
                    UnsavedChangesView(conflict: store.hasLibraryConflict, canReload: store.canReloadLibrary,
                                       retry: store.retrySave, reload: store.reloadLibrary)
                }
                RenameImageView(session: session) { store.rename(session.id, to: $0) }
                    .disabled(store.quitting)
            }.frame(width: 400)
        }
        .sheet(isPresented: $store.showArchive, onDismiss: sheetDidDismiss) { RecentlyDeletedView(store: store) }
        .task(id: store.notice?.id) {
            let id = store.notice?.id
            guard id != nil else { return }
            do { try await Task.sleep(for: .seconds(5)) }
            catch { return }
            if store.notice?.id == id { store.notice = nil }
        }
        .alert(StudioErrorTermination.title, isPresented: Binding(get: { error != nil }, set: { if !$0 { dismissError(error) } }), presenting: error) { presented in
            Button("好", role: .cancel) { dismissError(presented) }
            if store.hasUnsavedChanges {
                Button("重试保存") { store.retrySave(); dismissError(presented) }
            }
            if store.modelError && !store.busy {
                Button("重新加载模型") {
                    guard dismissError(presented) else { return }
                    Task { await store.loadModel() }
                }
            }
            SettingsLink { Text("打开设置") }
        } message: { Text($0.message) }
    }

    @discardableResult private func dismissError(_ shown: StudioErrorEvent?) -> Bool {
        guard errorPresentation.dismiss(shown) else { return false }
        return store.dismissError(shown)
    }

    private var workspace: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            LibraryView(store: store)
                .overlay(alignment: .trailing) {
                    WorkspaceResizeHandle(width: sidebarWidth, label: "资料库侧栏宽度", edge: .leading) {
                        sidebarWidth = max(210, min(300, $0))
                    }
                }
                .navigationSplitViewColumnWidth(sidebarWidth)
        } detail: {
            WorkspaceDetailView(store: store)
        }
        .navigationTitle("VQA Studio")
        .modifier(StudioToolbarMaterial())
        .toolbar {
            if #available(macOS 26.0, *) {
                ToolbarItem(placement: .navigation) {
                    Color.clear.frame(width: 8, height: 1).accessibilityHidden(true)
                }.sharedBackgroundVisibility(.hidden)
            } else {
                ToolbarItem(placement: .navigation) {
                    Color.clear.frame(width: 8, height: 1).accessibilityHidden(true)
                }
            }
            ToolbarItemGroup(placement: .primaryAction) {
                Button("导入图片", systemImage: "photo.badge.plus", action: store.chooseImage).help("导入图片 ⌘O")
                    .modifier(StudioControlStyle())
                    .disabled(!store.canImport)
                Menu("导出问答", systemImage: "square.and.arrow.up") {
                    Button("导出问答记录（JSON）…", action: store.exportSession)
                    Button("导出阅读文档（Markdown）…") { store.exportSession(readable: true) }
                    Divider()
                    Button("复制全部问答", action: store.copyConversation)
                }.disabled(store.selected == nil)
                Divider()
                Button(store.showInspector ? "隐藏问答面板" : "显示问答面板", systemImage: "sidebar.right") { store.showInspector.toggle() }
                    .modifier(StudioControlStyle())
                    .help("显示或隐藏问答面板 ⌥⌘I")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
