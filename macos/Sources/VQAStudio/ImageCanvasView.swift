import SwiftUI

struct ImageCanvasView: View {
    @Bindable var store: StudioStore
    @State private var targeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            WorkspaceHeader(
                title: store.selected?.title ?? "看见图片里的答案",
                subtitle: store.selected == nil ? "从一张图片开始，探索你关心的细节。" : "图片预览"
            ) { EmptyView() }
            Divider()

            GeometryReader { geometry in
                ZStack {
                    StudioTheme.canvas
                    if let image = store.preview.image {
                        if store.imageZoom != .fit {
                            ScrollView([.horizontal, .vertical]) {
                                Image(image.bitmap, scale: 1, label: Text(store.selected?.title ?? "当前图片")).resizable()
                                    .frame(width: max(1, Double(image.width) * store.imageZoom.rawValue), height: max(1, Double(image.height) * store.imageZoom.rawValue))
                                    .accessibilityLabel(store.selected?.title ?? "当前图片").padding(20)
                                    .frame(minWidth: geometry.size.width, minHeight: geometry.size.height)
                            }
                        } else {
                            Image(image.bitmap, scale: 1, label: Text(store.selected?.title ?? "当前图片")).resizable().scaledToFit()
                                .frame(width: max(1, geometry.size.width - 48), height: max(1, geometry.size.height - 48))
                                .accessibilityLabel(store.selected?.title ?? "当前图片")
                                .shadow(color: .black.opacity(0.08), radius: 16, y: 7)
                        }
                    } else if store.preview.loading { ProgressView("正在读取图片…") }
                    else if store.selected != nil {
                        MissingImageView(problem: store.preview.problem, reread: store.rereadImage, chooseImage: store.chooseImage)
                    }
                    else { EmptyCanvasView(store: store) }
                    if store.importing {
                        VStack(spacing: 14) {
                            ProgressView(store.importStatus)
                            Button(store.stoppingImport ? "停止中…" : "停止导入", action: store.cancelImport)
                                .disabled(store.stoppingImport || store.quitting)
                        }.padding(20).modifier(StudioGlassSurface())
                    }
                    if targeted { RoundedRectangle(cornerRadius: 12).strokeBorder(StudioTheme.accent, style: StrokeStyle(lineWidth: 2, dash: [6])).padding(6) }
                }
            }.clipShape(.rect(cornerRadius: 12))
                .overlay(alignment: .topTrailing) {
                    if store.selected != nil { ImageZoomControl(store: store).padding(12) }
                }
                .padding(.horizontal, 24).padding(.top, 24)
                .dropDestination(for: URL.self) { urls, _ in
                    let files = urls.filter(\.isFileURL)
                    guard !files.isEmpty, store.canImport else { return false }
                    store.importImages(files)
                    return true
                } isTargeted: { targeted = $0 }

            HStack {
                Label("图片保存在本机", systemImage: "lock").font(.caption).foregroundStyle(StudioTheme.secondary)
                Spacer()
                if let image = store.preview.image {
                    Text("\(image.width) × \(image.height)").font(.caption.monospacedDigit()).foregroundStyle(StudioTheme.secondary)
                }
            }.padding(.horizontal, 24).padding(.vertical, 20)
        }.background(.white)
            .overlay(alignment: .bottom) {
                if let notice = store.notice {
                    StudioNoticeView(notice: notice, undo: { store.undoNotice(notice) }) { store.notice = nil }
                }
            }
    }
}
