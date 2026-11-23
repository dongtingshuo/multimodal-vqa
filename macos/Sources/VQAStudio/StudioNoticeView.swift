import SwiftUI

struct StudioNoticeView: View {
    let notice: StudioNotice
    let undo: () -> Void
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Text(notice.message).font(.subheadline).fixedSize(horizontal: false, vertical: true)
            if notice.restoreID != nil || notice.draftUndoID != nil { Button("撤销", action: undo).buttonStyle(.bordered) }
            Button("关闭提示", systemImage: "xmark", action: dismiss)
                .labelStyle(.iconOnly).buttonStyle(.plain).frame(width: 20, height: 20)
        }
        .padding(12).modifier(StudioGlassSurface())
        .frame(maxWidth: 440).padding(.horizontal, 24).padding(.bottom, 64)
    }
}
