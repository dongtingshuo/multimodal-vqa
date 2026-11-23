import SwiftUI

struct ImageZoomControl: View {
    @Bindable var store: StudioStore

    var body: some View {
        Menu {
            Picker("图片缩放", selection: $store.imageZoom) {
                ForEach(ImageZoom.allCases) { zoom in Text(zoom.title).tag(zoom) }
            }
            Divider()
            Button("在访达中显示") { if let url = store.imageURL { NSWorkspace.shared.activateFileViewerSelecting([url]) } }
        } label: {
            Label(store.imageZoom.title, systemImage: "viewfinder")
                .font(.subheadline.weight(.medium))
        }
        .menuStyle(.borderlessButton).fixedSize()
        .padding(.horizontal, 12).padding(.vertical, 8).frame(minHeight: 36)
        .foregroundStyle(.primary)
        .modifier(StudioGlassSurface(cornerRadius: 100, interactive: true))
        .help("图片缩放与显示方式")
    }
}
