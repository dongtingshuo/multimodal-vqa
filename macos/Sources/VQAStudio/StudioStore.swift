import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable
final class StudioStore {
    nonisolated static var supportURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("VQA Studio")
    }
    var settings = AppSettings.initial()
    var sessions: [ImageSession] = []
    var archived: [ArchivedImage] = []
    var selectedID: UUID? {
        didSet {
            if oldValue != selectedID { imageZoom = .fit; refreshImagePreview() }
            scheduleSave()
        }
    }
    private var drafts = QuestionDrafts()
    var question: String {
        get { drafts[selectedID] }
        set { drafts[selectedID] = newValue; scheduleSave() }
    }
    var search = ""
    var sortOrder: LibrarySortOrder = .imported { didSet { scheduleSave() } }
    var favoritesOnly = false { didSet { scheduleSave() } }
    var favoriteIDs: Set<UUID> = []
    var busy = false
    var cancelling = false
    private var settingsCheck: Task<String?, Error>?
    var checkingModelSettings: Bool { settingsCheck != nil }
    private(set) var preparingImage = false
    var importing = false
    private(set) var stoppingImport = false
    private(set) var quitting = false
    private(set) var hasUnsavedChanges = false
    private(set) var hasLibraryConflict = false
    private(set) var libraryProblem: String?
    var importStatus = "正在导入图片…"
    var ready = false
    var pendingQuestion: PendingQuestion?
    private(set) var interruptedQuestions: [PendingQuestion] = []
    var status = "正在准备模型"
    var modelName = "本地模型"
    private(set) var errorEvent: StudioErrorEvent?
    private(set) var dismissingErrorForTermination = false
    var errorMessage: String? {
        get { errorEvent?.message }
        set { errorEvent = newValue.map { StudioErrorEvent(message: $0) } }
    }
    var presentedError: StudioErrorEvent? { quitting || dismissingErrorForTermination ? nil : errorEvent }
    var presentedErrorMessage: String? { presentedError?.message }
    var modelError = false
    let preview = ImagePreview()
    var imageZoom: ImageZoom = .fit
    var showInspector = true { didSet { scheduleSave() } }
    var showArchive = false
    var renameTarget: ImageSession?
    var notice: StudioNotice?
    var searchFocusRequest = 0
    var questionFocusRequest = 0
    private(set) var requestedFocus: WorkspaceFocus?
    var latestQuestionRequest = 0
    private(set) var canModifyLibrary = true
    @ObservationIgnored private let client: BackendClient
    @ObservationIgnored private let storage: LibraryStorage
    @ObservationIgnored private let importConverter: @Sendable (ImageImportRequest) async throws -> Data
    @ObservationIgnored private let imagePreparer: @Sendable (URL, ImageFileStamp) async throws -> PredictionImage
    @ObservationIgnored private let recoveryRequiresCurrent: Bool
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var saveErrorID: UUID?
    @ObservationIgnored private var modelErrorID: UUID?
    @ObservationIgnored private var importTask: Task<Void, Never>?
    @ObservationIgnored private var activeImportTask: Task<Data, Error>?
    @ObservationIgnored private var imagePreparationTask: Task<PredictionImage, Error>?
    @ObservationIgnored private var activePredictionImage: PredictionImage?
    @ObservationIgnored private var importQueue: [ImageImportRequest] = []
    @ObservationIgnored private var storageReady = false
    @ObservationIgnored private var started = false
    @ObservationIgnored private var startupTask: Task<Void, Never>?
    @ObservationIgnored private var draftReplacement: DraftReplacement?

    var selected: ImageSession? { sessions.first { $0.id == selectedID } }
    var imageURL: URL? { selected.map { imageURL(for: $0) } }
    var searchQuery: String { search.trimmingCharacters(in: .whitespacesAndNewlines) }
    var hasLibraryFilter: Bool { !searchQuery.isEmpty || favoritesOnly }
    var canImport: Bool { canModifyLibrary && !hasLibraryConflict && !quitting && !stoppingImport }
    var canReloadLibrary: Bool { (!canModifyLibrary || hasLibraryConflict) && !busy && !importing && !quitting }
    var filteredSessions: [ImageSession] {
        LibraryQuery.results(sessions, query: searchQuery, favoritesOnly: favoritesOnly, favorites: favoriteIDs, sort: sortOrder)
    }
    var isAnsweringSelectedImage: Bool { pendingQuestion != nil && pendingQuestion?.sessionID == selectedID }
    var recoverableQuestions: [PendingQuestion] { interruptedQuestions.filter { $0.sessionID == selectedID } }
    var questionStatus: String {
        if cancelling { return "正在停止…" }
        if preparingImage && isAnsweringSelectedImage { return "正在准备图片…" }
        if isAnsweringSelectedImage { return "正在生成候选答案…" }
        if pendingQuestion != nil { return "另一张图片正在分析中" }
        if preview.loading { return "正在读取图片…" }
        if preview.problem != nil { return "图片无法读取，请重新读取或导入" }
        return ready ? "本地模型已就绪" : "请加载模型后继续提问"
    }
    var canAsk: Bool {
        canModifyLibrary && !quitting && selected != nil && preview.image != nil && preview.url == imageURL
            && preview.stamp != nil
            && ready && !busy && !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    init(directory: URL = StudioStore.supportURL,
         imagePreparer: @escaping @Sendable (URL, ImageFileStamp) async throws -> PredictionImage = { try await PredictionImage.prepare(source: $0, stamp: $1) },
         importConverter: @escaping @Sendable (ImageImportRequest) async throws -> Data = { try await ImageImportClient().pngData(for: $0) }) {
        storage = LibraryStorage(directory: directory)
        self.importConverter = importConverter
        self.imagePreparer = imagePreparer
        recoveryRequiresCurrent = FileManager.default.fileExists(atPath: directory.appendingPathComponent("library.json").path)
        client = BackendClient(logDirectory: directory)
        do { restoreState(try storage.load(), recoveringInterrupted: true) }
        catch {
            canModifyLibrary = false
            libraryProblem = error.localizedDescription
        }
        storageReady = true
    }

    private var currentState: LibraryState {
        LibraryState(sessions: sessions, archived: archived, drafts: drafts.saved, favoriteIDs: favoriteIDs,
                     selectedID: selectedID, showInspector: showInspector, sortOrder: sortOrder, favoritesOnly: favoritesOnly,
                     interruptedQuestions: interruptedQuestions.isEmpty ? nil : interruptedQuestions)
    }

    private struct EditSnapshot {
        let library: LibraryState
        let search: String
        let zoom: ImageZoom
    }

    private var editSnapshot: EditSnapshot {
        EditSnapshot(library: currentState, search: search, zoom: imageZoom)
    }

    private func rollback(_ snapshot: EditSnapshot) {
        restoreState(snapshot.library)
        search = snapshot.search
        imageZoom = snapshot.zoom
    }

    private func restoreState(_ state: LibraryState, recoveringInterrupted: Bool = false) {
        sessions = state.sessions
        let activeIDs = Set(sessions.map(\.id))
        archived = state.archived.filter { !activeIDs.contains($0.id) }
        drafts = QuestionDrafts(saved: state.drafts)
        interruptedQuestions = state.interruptedQuestions ?? []
        if recoveringInterrupted {
            interruptedQuestions = []
            for submission in state.interruptedQuestions ?? [] { retainUnfinished(submission) }
        }
        favoriteIDs = state.favoriteIDs
        selectedID = state.selectedID.flatMap { activeIDs.contains($0) ? $0 : sessions.first?.id }
        showInspector = state.showInspector
        sortOrder = state.sortOrder
        favoritesOnly = state.favoritesOnly
    }

    private func scheduleSave() {
        guard storageReady, canModifyLibrary else { return }
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(350)) }
            catch { return }
            self?.persist()
        }
    }

    private func saveLibrary() throws {
        guard canModifyLibrary else { throw NSError(domain: "VQAStudio", code: 11, userInfo: [NSLocalizedDescriptionKey: "资料库保存已暂停，原文件不会被覆盖。"] ) }
        saveTask?.cancel()
        var state = currentState
        if let pendingQuestion, state.drafts[pendingQuestion.sessionID.uuidString] == nil {
            state.drafts[pendingQuestion.sessionID.uuidString] = pendingQuestion.question
        }
        if let pendingQuestion {
            var unfinished = state.interruptedQuestions ?? []
            if !unfinished.contains(where: { $0.id == pendingQuestion.id }) { unfinished.append(pendingQuestion) }
            state.interruptedQuestions = unfinished
        }
        do {
            try storage.save(state)
            hasLibraryConflict = false; hasUnsavedChanges = false
            if let saveErrorID, errorEvent?.id == saveErrorID { errorMessage = nil }
            saveErrorID = nil
        }
        catch {
            if error is LibraryConflictError { hasLibraryConflict = true }
            if !hasUnsavedChanges {
                let message = "资料库未能保存：\(error.localizedDescription)"
                errorMessage = message; saveErrorID = errorEvent?.id
            }
            hasUnsavedChanges = true
            modelError = false
            throw error
        }
    }

    @discardableResult private func persist() -> Bool {
        do { try saveLibrary(); return true }
        catch { return false }
    }

    func retrySave() {
        hasUnsavedChanges = false
        if persist() { notify("资料库已保存") }
    }

    func suspendErrorForTermination() -> Bool {
        guard presentedError != nil else { return false }
        dismissingErrorForTermination = true
        return true
    }

    @discardableResult func dismissError(_ presented: StudioErrorEvent?) -> Bool {
        guard !quitting, !dismissingErrorForTermination, let presented, errorEvent?.id == presented.id else { return false }
        errorEvent = nil
        return true
    }

    func openDataFolder() { NSWorkspace.shared.open(storage.directory) }
    func openLog() { NSWorkspace.shared.open(storage.directory.appendingPathComponent("engine.log")) }
    func refreshImagePreview() { preview.select(imageURL) }
    func rereadImage() { preview.select(imageURL, force: true) }

    func reloadLibrary() {
        guard canReloadLibrary else { return }
        do {
            let conflict = hasLibraryConflict
            if conflict { _ = try storage.saveRecoveryCopy(currentState) }
            let restored = try storage.reload(requireCurrent: recoveryRequiresCurrent || conflict)
            restoreState(restored, recoveringInterrupted: true)
            canModifyLibrary = true
            libraryProblem = nil
            hasUnsavedChanges = false
            hasLibraryConflict = false
            refreshImagePreview()
            errorMessage = nil
            notify(conflict ? "已将当前修改保存到资料库 Recovery 文件夹，并重新读取资料库。" : "资料库已重新读取")
        } catch {
            if canModifyLibrary { modelError = false; errorMessage = error.localizedDescription }
            else { libraryProblem = error.localizedDescription }
        }
    }

    func beginStartup() {
        guard startupTask == nil, !quitting else { return }
        // Startup belongs to the application, not the lifetime of its first visible window.
        startupTask = Task { await startup() }
    }

    private func startup() async {
        guard !started else { return }
        started = true
        let args = ProcessInfo.processInfo.arguments
        if let flag = args.firstIndex(of: "--open-image"), args.indices.contains(flag + 1) {
            importImages([URL(fileURLWithPath: args[flag + 1])])
            await importTask?.value
        }
        guard !Task.isCancelled, !quitting else { return }
        await loadModel()
        if let flag = args.firstIndex(of: "--ask"), args.indices.contains(flag + 1), ready, !Task.isCancelled, !quitting {
            await preview.waitUntilLoaded()
            question = args[flag + 1]
            await ask()
        }
    }

    func loadModel() async {
        await configureModel(with: settings, saveSettings: false)
    }

    @discardableResult
    private func configureModel(with updated: AppSettings, saveSettings: Bool) async -> String? {
        guard !busy, !quitting else { return nil }
        busy = true
        defer { busy = false; cancelling = false }

        let message: String?
        do {
            message = try await checkModelSettings(updated)
        } catch is CancellationError {
            if !quitting {
                if !ready { status = "模型已停止" }
                notify("已停止检查模型文件。")
            }
            return nil
        } catch {
            message = "无法检查模型文件，请重试。"
        }
        guard !quitting else { return nil }
        if let message {
            if !saveSettings {
                if !ready { status = "运行环境需要设置" }
                reportModelError(message)
            }
            return message
        }
        if saveSettings {
            let reload = needsModelReload(for: updated)
            settings = updated
            if let data = try? JSONEncoder().encode(updated) {
                UserDefaults.standard.set(data, forKey: "modelSettings")
            }
            if !reload {
                notify("候选答案数量已更新")
                return nil
            }
        }
        return await replaceModel(with: updated)
    }

    private func checkModelSettings(_ updated: AppSettings) async throws -> String? {
        let check = Task.detached(priority: .userInitiated) {
            try await SettingsValidationClient().message(for: updated)
        }
        settingsCheck = check
        defer { settingsCheck = nil }
        let message = try await withTaskCancellationHandler { try await check.value } onCancel: { check.cancel() }
        guard !cancelling, !quitting else { throw CancellationError() }
        try Task.checkCancellation()
        return message
    }

    private func replaceModel(with updated: AppSettings) async -> String? {
        ready = false; status = "正在加载模型…"
        await client.shutdownAndWait()
        do {
            guard !cancelling, !quitting else { throw CancellationError() }
            try Task.checkCancellation()
            let response = try await client.request(settings: updated, action: "load")
            guard !cancelling else { throw CancellationError() }
            modelName = response.model?.uppercased() ?? "VQA"
            status = "\(modelName) · \((response.device ?? updated.device).uppercased())"
            ready = true
            if let modelErrorID, errorEvent?.id == modelErrorID { errorMessage = nil }
            modelErrorID = nil
            modelError = false
        } catch is CancellationError {
            status = "模型已停止"; notify("已停止加载，可随时重新加载模型。")
        } catch {
            status = "模型尚未就绪"
            reportModelError(error.localizedDescription)
            return error.localizedDescription
        }
        return nil
    }

    private func reportModelError(_ message: String) {
        modelError = true
        errorMessage = message
        modelErrorID = errorEvent?.id
    }

    func ask() async {
        guard canAsk, let id = selectedID, let url = imageURL, let stamp = preview.stamp else { return }
        let submitted = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let submission = PendingQuestion(sessionID: id, question: submitted)
        question = ""
        pendingQuestion = submission
        // Persist before dispatch: a quick quit or crash must not lose a submitted question.
        guard persist() else {
            retainUnfinished(submission)
            pendingQuestion = nil
            return
        }
        busy = true
        preparingImage = true
        var requestedPrediction = false
        defer { busy = false; cancelling = false; preparingImage = false; imagePreparationTask = nil; pendingQuestion = nil }
        do {
            let imagePreparer = imagePreparer
            let preparation = Task.detached(priority: .userInitiated) { try await imagePreparer(url, stamp) }
            imagePreparationTask = preparation
            let input = try await withTaskCancellationHandler { try await preparation.value } onCancel: { preparation.cancel() }
            imagePreparationTask = nil; preparingImage = false
            activePredictionImage = input
            defer { input.remove(); activePredictionImage = nil }
            guard !cancelling, !quitting else { throw CancellationError() }
            try Task.checkCancellation()
            requestedPrediction = true
            let response = try await client.request(settings: settings, action: "predict", image: input.url.path, question: submitted)
            guard !cancelling else { throw CancellationError() }
            guard let answers = response.answers, !answers.isEmpty, let elapsed = response.elapsed else {
                throw NSError(domain: "VQAStudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "模型未返回候选答案，请重新尝试。"])
            }
            if let index = sessions.firstIndex(where: { $0.id == id }) {
                sessions[index].questions.append(QuestionRecord(question: submitted, answers: answers, elapsed: elapsed, model: response.model ?? modelName, checkpoint: response.checkpoint ?? "", device: response.device ?? settings.device))
                interruptedQuestions.removeAll { $0.sessionID == id && $0.question == submitted }
                pendingQuestion = nil
                persist()
            }
        } catch let error as ImageSnapshotError {
            retainUnfinished(submission)
            if selectedID == id { rereadImage() }
            pendingQuestion = nil; persist()
            modelError = false; errorMessage = error.localizedDescription
        } catch is CancellationError {
            retainUnfinished(submission)
            if requestedPrediction { ready = false; status = "模型已停止" }
            pendingQuestion = nil; persist()
            notify(requestedPrediction ? "已停止分析，未完成的问题已保留。" : "已停止准备图片，原问题已保留。")
        } catch {
            if sessions.contains(where: { $0.id == id }) { retainUnfinished(submission) }
            pendingQuestion = nil; persist()
            if (error as? BackendError)?.needsReload != false {
                ready = false; status = "模型连接已中断"
            }
            reportModelError(error.localizedDescription)
        }
    }

    func cancelModelTask() {
        guard busy, !cancelling else { return }
        cancelling = true
        if let imagePreparationTask { imagePreparationTask.cancel() }
        else if let settingsCheck { settingsCheck.cancel() }
        else { client.cancel() }
    }

    func chooseImage() {
        guard canImport else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]; panel.allowsMultipleSelection = true
        panel.prompt = "导入图片"
        StudioFilePanel.shared.present(panel) { [weak self] response in
            guard response == .OK, let self, !self.quitting else { return }
            self.importImages(panel.urls)
        }
    }

    func importImage(_ url: URL) { importImages([url]) }

    func importImages(_ urls: [URL]) {
        enqueueImports(urls.filter(\.isFileURL).map(ImageImportRequest.file))
    }

    private func enqueueImports(_ requests: [ImageImportRequest]) {
        guard canImport else { return }
        importQueue.append(contentsOf: requests)
        guard !importing, !importQueue.isEmpty else { return }
        importing = true
        importTask = Task {
            var completed = 0
            var failures: [String] = []
            defer { importing = false; stoppingImport = false; importTask = nil }
            while !importQueue.isEmpty && !stoppingImport {
                importStatus = "正在导入 \(completed + failures.count + 1) / \(completed + failures.count + importQueue.count)…"
                let request = importQueue.removeFirst()
                let converter = importConverter
                do {
                    let conversion = Task.detached(priority: .userInitiated) { try await converter(request) }
                    activeImportTask = conversion
                    defer { activeImportTask = nil }
                    let png = try await conversion.value
                    if stoppingImport { break }
                    try savePNG(png, title: request.title)
                    completed += 1
                } catch {
                    if stoppingImport { break }
                    failures.append("\(request.displayName)：\(error.localizedDescription)")
                    if hasLibraryConflict {
                        importQueue.removeAll()
                        break
                    }
                }
            }
            if stoppingImport { notify(completed > 0 ? "已停止导入，已导入的 \(completed) 张图片已保留。" : "已停止导入") }
            else if completed > 0 { notify("已导入 \(completed) 张图片") }
            if !failures.isEmpty {
                modelError = false
                errorMessage = hasLibraryConflict
                    ? "资料库文件已在外部更改，已暂停本批次剩余图片的导入。已成功导入的图片会保留。请点击“保存副本并重新读取”，恢复后可重新选择剩余图片。\n" + failures.prefix(3).joined(separator: "\n")
                    : "有 \(failures.count) 张图片未能导入，已成功导入的图片会保留。\n" + failures.prefix(3).joined(separator: "\n")
            }
        }
    }

    func cancelImport() {
        guard importing, !stoppingImport, !quitting else { return }
        stopImportQueue()
    }

    func stopImportsForQuit() {
        guard quitting, importing, !stoppingImport else { return }
        stopImportQueue()
    }

    private func stopImportQueue() {
        stoppingImport = true
        importQueue.removeAll()
        activeImportTask?.cancel()
        importStatus = "正在停止导入…"
    }

    func pasteImage() {
        pasteImage(from: .general)
    }

    func pasteImage(from pasteboard: NSPasteboard) {
        guard canImport else { return }
        if let type = pasteboard.availableType(from: [.png, .tiff]), let data = pasteboard.data(forType: type) {
            enqueueImports([.clipboard(data)])
        } else if let files = pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL], !files.isEmpty {
            importImages(files)
        } else {
            modelError = false; errorMessage = "剪贴板中没有可用图片。复制图片或访达中的图片文件后再试一次。"
        }
    }

    private func savePNG(_ png: Data, title: String) throws {
        let folder = storage.directory.appendingPathComponent("Images")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = UUID().uuidString + ".png"
        let url = folder.appendingPathComponent(name)
        try png.write(to: url, options: .atomic)
        let previous = editSnapshot
        let session = ImageSession(title: title, imageName: name)
        sessions.insert(session, at: 0); selectedID = session.id; imageZoom = .fit; search = ""; favoritesOnly = false
        do { try saveLibrary() }
        catch {
            rollback(previous)
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    func openExample() {
        let title = "COCO_val2014_000000000143"
        if let session = sessions.first(where: { $0.title == title }) {
            search = ""; favoritesOnly = false; selectedID = session.id
        } else {
            importImage(URL(fileURLWithPath: settings.project).appendingPathComponent("data/vqa/val2014/\(title).jpg"))
        }
    }

    func imageURL(for session: ImageSession) -> URL { storage.directory.appendingPathComponent("Images").appendingPathComponent(session.imageName) }

    func reveal(_ session: ImageSession) { NSWorkspace.shared.activateFileViewerSelecting([imageURL(for: session)]) }

    @discardableResult
    func rename(_ id: UUID, to title: String) -> Bool {
        let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canModifyLibrary, !quitting, !title.isEmpty, let index = sessions.firstIndex(where: { $0.id == id }) else { return false }
        let previous = editSnapshot
        sessions[index].title = title
        let saved = persist()
        if saved { notify("图片已重命名") } else { rollback(previous) }
        return saved
    }

    func toggleFavorite(_ id: UUID) {
        guard canModifyLibrary, !quitting else { return }
        let previous = editSnapshot
        if favoriteIDs.contains(id) { favoriteIDs.remove(id) } else { favoriteIDs.insert(id) }
        if !persist() { rollback(previous) }
    }

    func remove(_ id: UUID) {
        guard canModifyLibrary, !quitting, pendingQuestion?.sessionID != id,
              let session = sessions.first(where: { $0.id == id }) else { return }
        let previous = editSnapshot
        archived.insert(ArchivedImage(session: session), at: 0)
        sessions.removeAll { $0.id == id }
        if selectedID == id { selectedID = filteredSessions.first?.id }
        if persist() { notice = StudioNotice(message: "已移至最近删除", restoreID: id) }
        else { rollback(previous) }
    }

    func restoreArchived(_ id: UUID) {
        guard canModifyLibrary, !quitting, let archivedImage = archived.first(where: { $0.id == id }) else { return }
        let previous = editSnapshot
        sessions.insert(archivedImage.session, at: 0)
        archived.removeAll { $0.id == id }
        search = ""; favoritesOnly = false; selectedID = id
        if persist() { notify("图片与问答已恢复") } else { rollback(previous) }
    }

    func resetLibraryFilter() { search = ""; favoritesOnly = false }

    private func retainUnfinished(_ submission: PendingQuestion) {
        if drafts[submission.sessionID].isEmpty { drafts[submission.sessionID] = submission.question }
        else if drafts[submission.sessionID] != submission.question,
                !interruptedQuestions.contains(where: { $0.sessionID == submission.sessionID && $0.question == submission.question }) {
            interruptedQuestions.append(submission)
        }
    }

    func recoverQuestion(_ id: UUID) {
        guard canModifyLibrary, !quitting, !busy,
              let recovered = recoverableQuestions.first(where: { $0.id == id }) else { return }
        let previous = editSnapshot
        interruptedQuestions.removeAll { $0.id == id }
        let current = question
        question = recovered.question
        let hasCurrentDraft = !current.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasCurrentDraft && current != recovered.question {
            retainUnfinished(PendingQuestion(sessionID: recovered.sessionID, question: current))
        }
        if persist() {
            focusQuestion()
            notify(hasCurrentDraft ? "已切换问题，上一份草稿仍保留在待恢复列表中。" : "未完成的问题已恢复到输入框")
        } else { rollback(previous) }
    }
    func focusSearch() { requestedFocus = .search; searchFocusRequest += 1 }
    func focusQuestion() { requestedFocus = .question; showInspector = true; questionFocusRequest += 1 }
    func showLatestQuestion() { showInspector = true; latestQuestionRequest += 1 }
    func reuseQuestion(_ text: String) {
        guard canModifyLibrary, !quitting else { return }
        draftReplacement = nil
        if !question.isEmpty && question != text {
            let undo = DraftReplacement(sessionID: selectedID, previous: question, replacement: text)
            draftReplacement = undo
            notice = StudioNotice(message: "已填入问题，可撤销恢复上一份草稿。", draftUndoID: undo.id)
        }
        question = text
        focusQuestion()
    }

    func undoNotice(_ notice: StudioNotice) {
        if let id = notice.restoreID { restoreArchived(id); return }
        guard canModifyLibrary, !quitting, let undo = draftReplacement, undo.id == notice.draftUndoID else { return }
        draftReplacement = nil
        guard drafts[undo.sessionID] == undo.replacement else {
            notify("草稿已有新修改，已保留当前内容。")
            return
        }
        drafts[undo.sessionID] = undo.previous
        if persist() { notify("已恢复上一份草稿") }
    }
    func notify(_ message: String) { notice = StudioNotice(message: message) }

    func needsModelReload(for updated: AppSettings) -> Bool {
        !ready || updated.normalized == settings.normalized || !updated.hasSameRuntime(as: settings)
    }

    @discardableResult
    func apply(_ updated: AppSettings) async -> String? {
        await configureModel(with: updated.normalized, saveSettings: true)
    }

    func exportSession() { exportSession(readable: false) }

    func exportSession(readable: Bool) {
        guard let session = selected else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = readable ? [UTType(filenameExtension: "md") ?? .plainText] : [.json]
        panel.nameFieldStringValue = ConversationExporter.filename(for: session, readable: readable)
        StudioFilePanel.shared.present(panel) { [weak self] response in
            guard response == .OK, let self, !self.quitting, let url = panel.url else { return }
            do {
                if readable { try ConversationExporter.markdown(session).write(to: url, atomically: true, encoding: .utf8) }
                else {
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
                    try encoder.encode(session).write(to: url, options: .atomic)
                }
                self.notify("问答已导出")
            } catch { self.modelError = false; self.errorMessage = "导出失败：\(error.localizedDescription)" }
        }
    }

    func copyConversation() {
        guard let session = selected else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(ConversationExporter.markdown(session), forType: .string)
        notify("已复制全部问答")
    }

    func shutdown() {
        startupTask?.cancel()
        settingsCheck?.cancel()
        imagePreparationTask?.cancel()
        activePredictionImage?.remove()
        saveTask?.cancel()
        if canModifyLibrary { persist() }
        client.shutdown()
        preview.cancel()
    }

    func prepareToTerminate() async -> Bool {
        guard !quitting else { return false }
        quitting = true
        dismissingErrorForTermination = false
        if importing { notify("正在完成图片导入并保存，请稍候…") }
        await importTask?.value
        saveTask?.cancel()
        if canModifyLibrary && !persist() {
            errorMessage = hasLibraryConflict
                ? "资料库文件已在外部更改，已暂缓退出。请使用“保存副本并重新读取”保留当前修改并恢复资料库后，再退出。"
                : "资料库尚未保存，已暂缓退出以保留当前问答和草稿。请检查磁盘空间或文件夹权限后，点击“重试保存”再退出。"
            saveErrorID = errorEvent?.id
            quitting = false
            return false
        }
        startupTask?.cancel()
        if let settingsCheck {
            settingsCheck.cancel()
            _ = try? await settingsCheck.value
        }
        if let preparation = imagePreparationTask {
            preparation.cancel()
            if let image = try? await preparation.value { image.remove() }
        }
        await client.shutdownAndWait()
        activePredictionImage?.remove()
        await preview.cancelAndWait()
        return true
    }
}
