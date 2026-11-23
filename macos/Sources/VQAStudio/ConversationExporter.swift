import Foundation

enum ConversationExporter {
    static func filename(for session: ImageSession, readable: Bool) -> String {
        let title = session.title.components(separatedBy: CharacterSet(charactersIn: "/:\r\n\0")).joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var safeTitle = ""
        for character in title {
            guard safeTitle.utf8.count + String(character).utf8.count <= 180 else { break }
            safeTitle.append(character)
        }
        return "\(safeTitle.isEmpty ? "图片" : safeTitle)-问答.\(readable ? "md" : "json")"
    }

    static func markdown(_ session: ImageSession) -> String {
        let title = normalizedLines(session.title).replacingOccurrences(of: "\n", with: " ")
        var lines = ["# \(title)", "", "共 \(session.questions.count) 条问答", ""]
        for (index, record) in session.questions.enumerated() {
            lines += ["## 问题 \(index + 1)", "", "> " + normalizedLines(record.question).replacingOccurrences(of: "\n", with: "\n> "), ""]
            lines += ["| 候选答案 | 分数 |", "| --- | ---: |"]
            lines += record.answers.map { "| \(tableText($0.answer)) | \(String(format: "%.1f%%", $0.score * 100)) |" }
            lines += ["", "模型：\(record.model.uppercased()) · 设备：\(record.device.uppercased()) · 用时：\(String(format: "%.2f", record.elapsed)) 秒", ""]
        }
        lines += ["---", "候选答案受模型词表限制，分数不代表正确率。", ""]
        return lines.joined(separator: "\n")
    }

    private static func tableText(_ value: String) -> String {
        normalizedLines(value).replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
    }

    private static func normalizedLines(_ value: String) -> String {
        value.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
    }
}
