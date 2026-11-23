import SwiftUI

struct EmptyCanvasView: View {
    let store: StudioStore
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "photo.on.rectangle.angled").font(.system(size: 54, weight: .ultraLight)).foregroundStyle(StudioTheme.secondary).accessibilityHidden(true)
            VStack(spacing: 8) {
                Text("把图片拖到这里").font(.title2.weight(.medium))
                Text("支持 JPEG、PNG、HEIC 等常见图片格式")
                    .font(.subheadline).foregroundStyle(StudioTheme.secondary).multilineTextAlignment(.center)
            }
            Button("选择图片…", action: store.chooseImage).modifier(StudioControlStyle(prominent: true)).controlSize(.large).disabled(!store.canImport)
            Button("使用项目示例图片", action: store.openExample).buttonStyle(.link).disabled(!store.canImport)
            Text("也可以按 ⇧⌘V 粘贴图片").font(.caption).foregroundStyle(StudioTheme.secondary)
        }.padding(28).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
