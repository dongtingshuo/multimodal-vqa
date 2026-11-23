import SwiftUI

struct LibraryRow: View {
    let session: ImageSession
    let imageURL: URL
    var favorite = false
    var body: some View {
        HStack(spacing: 11) {
            LibraryThumbnailView(url: imageURL)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 4) {
                    Text(session.title).font(.subheadline.weight(.medium)).lineLimit(1).help(session.title)
                    if favorite { Image(systemName: "star.fill").font(.caption2).accessibilityLabel("已收藏") }
                }
                Text(session.questions.isEmpty ? "尚未提问" : "\(session.questions.count) 条问答")
                    .font(.caption)
            }
        }.padding(.vertical, 6)
    }
}
