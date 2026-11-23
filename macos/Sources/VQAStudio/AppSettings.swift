import Darwin
import Foundation

struct AppSettings: Codable, Equatable, Sendable {
    var project: String
    var python: String
    var config: String
    var checkpoint: String
    var device = "cpu"
    var offline = true
    var topk = 5

    static func initial() -> AppSettings {
        if let data = UserDefaults.standard.data(forKey: "modelSettings"),
           let value = try? JSONDecoder().decode(Self.self, from: data) { return value.normalized }
        if let url = Bundle.main.url(forResource: "Defaults", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let value = try? JSONDecoder().decode(Self.self, from: data) { return value.normalized }
        return Self(project: "", python: "/usr/bin/python3", config: "configs/default.yaml", checkpoint: "checkpoints/best.pt")
    }

    var workerSettings: [String: Any] {
        ["config_path": absolute(config), "checkpoint_path": absolute(checkpoint), "device": device, "offline": offline]
    }

    var normalized: Self {
        var value = self
        value.project = NSString(string: project.trimmingCharacters(in: .whitespacesAndNewlines)).expandingTildeInPath
        value.python = NSString(string: python.trimmingCharacters(in: .whitespacesAndNewlines)).expandingTildeInPath
        value.config = config.trimmingCharacters(in: .whitespacesAndNewlines)
        value.checkpoint = checkpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        return value
    }

    func hasSameRuntime(as other: Self) -> Bool {
        var comparable = normalized
        let previous = other.normalized
        // Candidate count is sent with each prediction and does not construct the model.
        comparable.topk = previous.topk
        return comparable == previous
    }

    var validationMessage: String? {
        let value = normalized
        let root = value.project
        guard !root.isEmpty, root.hasPrefix("/"),
              accessibleRegularFile(URL(fileURLWithPath: root).appendingPathComponent("vqa_project/desktop_worker.py").path) else {
            return "请选择包含 vqa_project 的项目文件夹。"
        }
        guard value.python.hasPrefix("/"), accessibleRegularFile(value.python, mode: R_OK | X_OK) else {
            return "请选择可执行的 Python 环境文件。"
        }
        guard !value.config.isEmpty, accessibleRegularFile(value.absolute(value.config)) else { return "请选择可读取的普通配置文件。" }
        guard !value.checkpoint.isEmpty, accessibleRegularFile(value.absolute(value.checkpoint)) else { return "请选择可读取的普通模型权重文件。" }
        guard ["cpu", "mps", "auto"].contains(value.device), (1...10).contains(value.topk) else { return "请选择支持的计算设备，并将候选答案数量设为 1–10。" }
        return nil
    }

    private func accessibleRegularFile(_ path: String, mode: Int32 = R_OK) -> Bool {
        // Follow environment symlinks, but never accept a pipe or device as a model input.
        URL(fileURLWithPath: path).withUnsafeFileSystemRepresentation { representation in
            guard let representation else { return false }
            var attributes = stat()
            return Darwin.fstatat(AT_FDCWD, representation, &attributes, 0) == 0
                && attributes.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG)
                && Darwin.access(representation, mode) == 0
        }
    }

    func absolute(_ path: String) -> String {
        let expanded = NSString(string: path.trimmingCharacters(in: .whitespacesAndNewlines)).expandingTildeInPath
        return expanded.hasPrefix("/") ? expanded : URL(fileURLWithPath: normalized.project).appendingPathComponent(expanded).path
    }
}
