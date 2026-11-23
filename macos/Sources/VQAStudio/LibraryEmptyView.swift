import SwiftUI

struct LibraryEmptyView: View {
    let searching: Bool
    var favoritesOnly = false
    let clearSearch: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: searching ? "magnifyingglass" : "photo.on.rectangle")
                .font(.system(size: 24)).accessibilityHidden(true)
            Text(searching ? (favoritesOnly ? "没有匹配的收藏" : "没有匹配的图片") : "资料库还没有图片")
                .font(.subheadline.weight(.medium)).foregroundStyle(.primary)
            Text(searching ? (favoritesOnly ? "可以在图片右键菜单中收藏，或清除筛选查看全部图片。" : "换个关键词，或清除搜索查看全部图片。") : "导入图片后，图片和问答会保存在这里。")
                .font(.caption).fixedSize(horizontal: false, vertical: true)
            if searching { Button(favoritesOnly ? "清除筛选" : "清除搜索", action: clearSearch).buttonStyle(.bordered) }
        }
        .foregroundStyle(StudioTheme.secondary).multilineTextAlignment(.center)
        .frame(maxWidth: .infinity).padding(.horizontal, 20).padding(.top, 32)
    }
}
