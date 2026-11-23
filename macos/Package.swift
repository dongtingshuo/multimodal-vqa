// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VQAStudio",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "VQAStudio", targets: ["VQAStudio"])],
    targets: [.executableTarget(name: "VQAStudio")]
)
