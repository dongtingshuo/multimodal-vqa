import SwiftUI

struct RecentlyDeletedView: View {
    @Bindable var store: StudioStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                Label("最近删除", systemImage: "trash").font(.title2.weight(.semibold))
                Text("图片、问答和草稿都可以恢复，不会自动清空。")
                    .font(.subheadline).foregroundStyle(StudioTheme.secondary)
            }.padding(24)
            Divider()
            if store.quitting { QuitProgressView(store: store) }
            if store.hasUnsavedChanges {
                UnsavedChangesView(conflict: store.hasLibraryConflict, canReload: store.canReloadLibrary,
                                   retry: store.retrySave, reload: store.reloadLibrary)
            }
            if store.archived.isEmpty {
                ContentUnavailableView("没有最近删除的图片", systemImage: "trash", description: Text("从资料库移除的图片会保留在这里。"))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(store.archived) { item in
                            HStack(spacing: 16) {
                                LibraryRow(session: item.session, imageURL: store.imageURL(for: item.session), favorite: store.favoriteIDs.contains(item.id))
                                Spacer(minLength: 8)
                                Text(item.removedAt, style: .date).font(.caption).foregroundStyle(StudioTheme.secondary)
                                Button("恢复") { store.restoreArchived(item.id) }.buttonStyle(.bordered)
                                    .disabled(!store.canModifyLibrary || store.quitting)
                                    .accessibilityLabel("恢复\(item.session.title)")
                            }.padding(.vertical, 4)
                                .accessibilityElement(children: .contain)
                        }
                    }.padding(.horizontal, 16).padding(.vertical, 8)
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Divider()
            HStack {
                Text("共 \(store.archived.count) 张图片").font(.caption).foregroundStyle(StudioTheme.secondary)
                Spacer()
                Button("完成") { dismiss() }.keyboardShortcut(.cancelAction)
                    .disabled(store.quitting)
            }.padding(20)
        }.frame(width: 640, height: 440)
    }
}
