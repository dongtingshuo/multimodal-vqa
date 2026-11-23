import Foundation

/// Separate bounds keep rows from starving the selected preview; cancellation also removes queued work.
actor ImageDecodingService {
    static let shared = ImageDecodingService()
    static let thumbnails = ImageDecodingService(maximumConcurrent: 2, maximumPixelSize: 84)
    private let client: ImageImportClient
    private let limiter: ImageWorkLimiter
    private let maximumPixelSize: Int?
    private var stopping = false
    private var operations: [UUID: Task<(image: DecodedImage, stamp: ImageFileStamp), Error>] = [:]

    init(client: ImageImportClient = ImageImportClient(timeout: .seconds(15)), maximumConcurrent: Int = 1, maximumPixelSize: Int? = nil) {
        self.client = client; limiter = ImageWorkLimiter(limit: maximumConcurrent)
        self.maximumPixelSize = maximumPixelSize
    }

    func decode(_ url: URL, cached: DecodedImage? = nil, stamp: ImageFileStamp? = nil) async throws -> (image: DecodedImage, stamp: ImageFileStamp) {
        guard !stopping else { throw CancellationError() }
        try Task.checkCancellation()
        let client = client, maximum = maximumPixelSize, limiter = limiter
        let id = UUID()
        let operation = Task {
            try await limiter.run {
                let result = try await client.readImage(at: url, maximumPixelSize: maximum, knownStamp: cached == nil ? nil : stamp)
                try Task.checkCancellation()
                if result.receipt.unchanged, let cached { return (image: cached, stamp: result.receipt.stamp) }
                guard let data = result.data else { throw ImageReadError.changed }
                let image = try ImageCodec.decode(data)
                try Task.checkCancellation()
                return (image: image, stamp: result.receipt.stamp)
            }
        }
        operations[id] = operation
        defer { operations[id] = nil }
        return try await withTaskCancellationHandler { try await operation.value } onCancel: { operation.cancel() }
    }

    func shutdown() async {
        stopping = true
        let pending = Array(operations.values)
        for task in pending { task.cancel() }
        for task in pending { _ = try? await task.value }
    }
}
