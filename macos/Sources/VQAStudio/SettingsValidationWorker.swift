import Darwin
import Foundation

enum SettingsValidationWorker {
    static let argument = "--vqa-settings-validation-worker"

    static func runIfRequested() {
        guard CommandLine.arguments.dropFirst().first == argument else { return }
        guard CommandLine.arguments.count == 3 else { Darwin.exit(64) }
        let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        do {
            let request = try Data(contentsOf: directory.appendingPathComponent("request.json"))
            let settings = try JSONDecoder().decode(AppSettings.self, from: request)
            let response = try JSONEncoder().encode(settings.validationMessage)
            try response.write(to: directory.appendingPathComponent("response.json"), options: .atomic)
            Darwin.exit(0)
        } catch {
            Darwin.exit(1)
        }
    }
}
