import SwiftUI

struct VQAStudioApp: App {
    @NSApplicationDelegateAdaptor(StudioAppDelegate.self) private var appDelegate
    @State private var store = StudioStore()
    @State private var quitAfterSheetDismissal = false
    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        Window("VQA Studio", id: "workspace") {
            WorkspaceView(store: store, sheetDidDismiss: workspaceSheetDidDismiss)
                .preferredColorScheme(.light)
                .task {
                    appDelegate.showWorkspace = { openWindow(id: "workspace") }
                    appDelegate.prepareToTerminate = { await store.prepareToTerminate() }
                    appDelegate.finishSheetTermination = finishPreparedSheetTermination
                    if let application = NSApplication.shared as? StudioApplication {
                        application.handleErrorTermination = {
                            StudioErrorTermination.shared.request(dismiss: store.suspendErrorForTermination) {
                                appDelegate.requestSheetTermination()
                            }
                        }
                        application.handleWorkspaceSheetTermination = {
                            guard store.showArchive || store.renameTarget != nil else { return false }
                            appDelegate.requestSheetTermination()
                            return true
                        }
                    }
                    appDelegate.importImages = store.importImages
                    store.beginStartup()
                }
        }
        .defaultSize(width: 1380, height: 870)
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(before: .windowArrangement) {
                Button("显示主窗口") { openWindow(id: "workspace") }.keyboardShortcut("1")
            }
            CommandGroup(replacing: .newItem) {
                Button("导入图片…") { performInWorkspace(store.chooseImage) }.keyboardShortcut("o").disabled(!store.canImport)
                Button("粘贴图片") { performInWorkspace(store.pasteImage) }.keyboardShortcut("v", modifiers: [.command, .shift]).disabled(!store.canImport)
                Button("停止导入", action: store.cancelImport).disabled(!store.importing || store.stoppingImport || store.quitting)
                Divider()
                Button("导出当前问答…") { performInWorkspace(store.exportSession) }.keyboardShortcut("e", modifiers: [.command, .shift]).disabled(store.selected == nil)
                Button("导出阅读文档…") { performInWorkspace { store.exportSession(readable: true) } }.disabled(store.selected == nil)
                Divider()
                Button("最近删除…") { performInWorkspace { store.showArchive = true } }
                Button("打开资料库文件夹", action: store.openDataFolder)
                if !store.canModifyLibrary { Button("重新读取资料库", action: store.reloadLibrary).disabled(!store.canReloadLibrary) }
            }
            CommandGroup(after: .textEditing) {
                Button("搜索资料库") { performInWorkspace(store.focusSearch) }.keyboardShortcut("f")
            }
            CommandMenu("问答") {
                Button("跳到最新问答") { performInWorkspace(store.showLatestQuestion) }
                    .keyboardShortcut(.downArrow, modifiers: [.command, .option])
                    .disabled(store.selected?.questions.isEmpty != false && !store.isAnsweringSelectedImage)
                Button("输入问题") { performInWorkspace(store.focusQuestion) }.keyboardShortcut("l")
                Button("提交问题") { Task { await store.ask() } }.keyboardShortcut(.return, modifiers: .command).disabled(!store.canAsk)
                Button("停止当前任务", action: store.cancelModelTask).keyboardShortcut(".").disabled(!store.busy || store.cancelling)
                Button("重新加载模型") { Task { await store.loadModel() } }.disabled(store.busy)
            }
            CommandMenu("图片") {
                Button("重命名…") { performInWorkspace { store.renameTarget = store.selected } }
                    .disabled(store.selected == nil || !store.canModifyLibrary || store.quitting)
                Button(store.selectedID.map { store.favoriteIDs.contains($0) } == true ? "取消收藏" : "收藏图片") {
                    if let id = store.selectedID { store.toggleFavorite(id) }
                }.disabled(store.selected == nil || !store.canModifyLibrary || store.quitting)
                Button("在访达中显示") { if let session = store.selected { store.reveal(session) } }.disabled(store.selected == nil)
                Divider()
                Button("移至最近删除") { if let id = store.selectedID { store.remove(id) } }
                    .disabled(store.selected == nil || !store.canModifyLibrary || store.quitting || store.isAnsweringSelectedImage)
            }
            CommandGroup(after: .sidebar) {
                Button(store.showInspector ? "隐藏问答面板" : "显示问答面板") { store.showInspector.toggle() }
                    .keyboardShortcut("i", modifiers: [.command, .option])
                Divider()
                Button("放大图片") { store.imageZoom = store.imageZoom.larger }.keyboardShortcut("=")
                    .disabled(store.selected == nil || store.imageZoom == .quadruple)
                Button("缩小图片") { store.imageZoom = store.imageZoom.smaller }.keyboardShortcut("-")
                    .disabled(store.selected == nil || store.imageZoom == .quarter)
                Button("图片适合窗口") { store.imageZoom = .fit }.keyboardShortcut("0", modifiers: [.command, .option])
                    .disabled(store.selected == nil)
                Button("图片原始尺寸") { store.imageZoom = .actual }.keyboardShortcut("1", modifiers: [.command, .option])
                    .disabled(store.selected == nil)
                Divider()
                Button("紧凑窗口") { resizeWorkspace(width: 1000, height: 680) }
                    .keyboardShortcut("0", modifiers: [.command, .shift])
                Button("标准窗口") { resizeWorkspace(width: 1380, height: 870) }
                    .keyboardShortcut("0", modifiers: .command)
            }
        }
        Settings { SettingsView(store: store).preferredColorScheme(.light) }
    }

    private func finishPreparedSheetTermination() {
        if store.showArchive || store.renameTarget != nil {
            quitAfterSheetDismissal = true
            store.showArchive = false
            store.renameTarget = nil
        } else {
            (NSApplication.shared as? StudioApplication)?.completePreparedTermination()
        }
    }

    private func workspaceSheetDidDismiss() {
        guard quitAfterSheetDismissal else { return }
        quitAfterSheetDismissal = false
        (NSApplication.shared as? StudioApplication)?.completePreparedTermination()
    }

    private func resizeWorkspace(width: CGFloat, height: CGFloat) {
        guard let window = NSApplication.shared.windows.first(where: { $0.identifier?.rawValue == "workspace" }) else {
            openWindow(id: "workspace")
            return
        }
        window.setContentSize(NSSize(width: width, height: height))
        window.makeKeyAndOrderFront(nil)
    }

    private func performInWorkspace(_ action: @MainActor () -> Void) {
        guard !store.quitting else { return }
        openWindow(id: "workspace")
        action()
    }
}
