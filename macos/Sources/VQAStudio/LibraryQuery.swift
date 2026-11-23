import Foundation

enum LibraryQuery {
    static func results(_ sessions: [ImageSession], query: String, favoritesOnly: Bool, favorites: Set<UUID>, sort: LibrarySortOrder) -> [ImageSession] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return sessions.filter { session in
            (!favoritesOnly || favorites.contains(session.id)) && (query.isEmpty || session.title.localizedStandardContains(query)
                || session.questions.contains { record in
                    record.question.localizedStandardContains(query) || record.answers.contains { $0.answer.localizedStandardContains(query) }
                })
        }.sorted { first, second in
            switch sort {
            case .name:
                let comparison = first.title.localizedStandardCompare(second.title)
                return comparison == .orderedSame ? first.id.uuidString < second.id.uuidString : comparison == .orderedAscending
            case .imported:
                return first.date == second.date ? first.id.uuidString < second.id.uuidString : first.date > second.date
            case .recentQuestions:
                let firstDate = first.questions.last?.date ?? first.date
                let secondDate = second.questions.last?.date ?? second.date
                return firstDate == secondDate ? first.id.uuidString < second.id.uuidString : firstDate > secondDate
            }
        }
    }
}
