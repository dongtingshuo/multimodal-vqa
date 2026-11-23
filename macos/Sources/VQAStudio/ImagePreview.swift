import Foundation
import Observation

@MainActor @Observable
final class ImagePreview {
    private(set) var url: URL?
    private(set) var image: DecodedImage?
    private(set) var loading = false
    private(set) var problem: String?
    @ObservationIgnored private(set) var stamp: ImageFileStamp?
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private let service: ImageDecodingService

    init(service: ImageDecodingService = .shared) { self.service = service }

    func select(_ selectedURL: URL?, force: Bool = false) {
        if !force, selectedURL == url, loading { return }
        let retainedImage = !force && selectedURL == url ? image : nil
        let retainedStamp = retainedImage == nil ? nil : stamp
        generation += 1
        task?.cancel()
        url = selectedURL; image = retainedImage; stamp = retainedStamp; problem = nil
        guard let selectedURL else { loading = false; task = nil; return }
        loading = retainedImage == nil
        let request = generation
        let service = service
        task = Task { [weak self] in
            do {
                let result = try await service.decode(selectedURL, cached: retainedImage, stamp: retainedStamp)
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.image = result.image; self.stamp = result.stamp; self.loading = false; self.task = nil
            } catch {
                guard !Task.isCancelled, let self, self.generation == request else { return }
                self.image = nil; self.stamp = nil
                self.problem = "暂时无法显示这张图片。\n\(error.localizedDescription)"
                self.loading = false; self.task = nil
            }
        }
    }

    func waitUntilLoaded() async {
        while let current = task {
            await current.value
            if Task.isCancelled { return }
        }
    }

    func cancel() {
        generation += 1; task?.cancel(); task = nil; loading = false
    }

    func cancelAndWait() async {
        let pending = task
        cancel()
        await pending?.value
    }
}
