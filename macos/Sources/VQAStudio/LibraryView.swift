import SwiftUI

struct LibraryView: View {
    @Bindable var store: StudioStore

    var body: some View {
        let filtered = store.filteredSessions
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable().scaledToFit().frame(width: 32, height: 32)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 3) {
                    Text("VQA Studio").font(.headline)
                    Text("视觉问答工作台").font(.caption).foregroundStyle(StudioTheme.secondary)
                }
                Spacer()
            }.padding(.horizontal, 16).padding(.top, 22).padding(.bottom, 24)

            Button(action: store.chooseImage) {
                Label("导入图片", systemImage: "plus").frame(maxWidth: .infinity).padding(.vertical, 5)
            }.modifier(StudioControlStyle(prominent: true)).padding(.horizontal, 16).disabled(!store.canImport)

            LibrarySearchField(text: $store.search, focusRequest: store.requestedFocus == .search ? store.searchFocusRequest : 0).frame(height: 24).padding(16)

            HStack {
                Text("图片资料库").font(.subheadline.weight(.semibold))
                Spacer()
                Text(!store.hasLibraryFilter ? "\(store.sessions.count)" : "\(filtered.count) / \(store.sessions.count)")
                    .font(.caption.monospacedDigit()).foregroundStyle(StudioTheme.secondary)
                    .accessibilityLabel(!store.hasLibraryFilter ? "共 \(store.sessions.count) 张图片" : "找到 \(filtered.count) 张，共 \(store.sessions.count) 张")
                Menu("排序与筛选", systemImage: "line.3.horizontal.decrease.circle") {
                    Picker("排序", selection: $store.sortOrder) {
                        ForEach(LibrarySortOrder.allCases) { order in Text(order.title).tag(order) }
                    }
                    Divider()
                    Toggle("只看收藏", isOn: $store.favoritesOnly)
                }.labelStyle(.iconOnly).menuStyle(.borderlessButton).menuIndicator(.hidden)
                    .fixedSize().help("排序与筛选")
            }.padding(.horizontal, 16).padding(.bottom, 8)

            List(selection: $store.selectedID) {
                ForEach(filtered) { session in
                    LibraryRow(session: session, imageURL: store.imageURL(for: session), favorite: store.favoriteIDs.contains(session.id)).tag(session.id)
                        .contextMenu {
                            Button("重命名…") { store.renameTarget = session }.disabled(!store.canModifyLibrary)
                            Button(store.favoriteIDs.contains(session.id) ? "取消收藏" : "收藏图片") { store.toggleFavorite(session.id) }
                                .disabled(!store.canModifyLibrary)
                            Button("在访达中显示") { store.reveal(session) }
                            Divider()
                            Button("移至最近删除", role: .destructive) { store.remove(session.id) }
                                .disabled(!store.canModifyLibrary || store.pendingQuestion?.sessionID == session.id)
                        }
                }
            }.listStyle(.sidebar)
                .overlay(alignment: .top) {
                    if filtered.isEmpty {
                        LibraryEmptyView(searching: store.hasLibraryFilter, favoritesOnly: store.favoritesOnly) { store.resetLibraryFilter() }
                    }
                }
            Spacer(minLength: 0)
            Button { store.showArchive = true } label: {
                HStack {
                    Label("最近删除", systemImage: "trash")
                    Spacer()
                    Text("\(store.archived.count)").monospacedDigit()
                }.font(.caption).foregroundStyle(StudioTheme.secondary)
            }.buttonStyle(.plain).padding(.horizontal, 16).padding(.vertical, 12)
            Divider()
            HStack(spacing: 9) {
                Image(systemName: store.ready ? "checkmark.circle.fill" : "circle.dotted")
                    .foregroundStyle(store.ready ? .green : StudioTheme.secondary)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.status).font(.caption.weight(.medium)).lineLimit(1)
                    Text("本地运行 · 历史保存在本机").font(.caption2).foregroundStyle(StudioTheme.secondary)
                }
                Spacer(minLength: 0)
                SettingsLink { Label("设置", systemImage: "gearshape").labelStyle(.iconOnly).frame(width: 20, height: 20) }.buttonStyle(.plain).help("模型与应用设置")
            }.padding(16)
        }
    }
}
