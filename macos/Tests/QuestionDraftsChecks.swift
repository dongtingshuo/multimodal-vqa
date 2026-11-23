import Foundation

@main
enum QuestionDraftsChecks {
    static func main() {
        checkIndependentDrafts()
        checkNewerDraftSurvivesFailure()
        checkFailureRestoresOriginalImage()
        checkRemovalKeepsOtherDrafts()
        print("4 question-draft regression checks passed")
    }

    static func checkIndependentDrafts() {
        let first = UUID(), second = UUID()
        var drafts = QuestionDrafts()
        drafts[first] = "What color is the bird?"
        precondition(drafts[second].isEmpty, "A different image must start with its own draft")
        drafts[second] = "How many people are there?"
        precondition(drafts[first] == "What color is the bird?")
        precondition(drafts[second] == "How many people are there?")
    }

    static func checkNewerDraftSurvivesFailure() {
        let image = UUID()
        var drafts = QuestionDrafts()
        drafts[image] = "A new question typed while waiting"
        drafts.restore("The submitted question", for: image)
        precondition(drafts[image] == "A new question typed while waiting", "Failure must not replace newer typing")
    }

    static func checkFailureRestoresOriginalImage() {
        let first = UUID(), second = UUID()
        var drafts = QuestionDrafts()
        drafts[second] = "Question for a different image"
        drafts.restore("Failed question for the first image", for: first)
        precondition(drafts[first] == "Failed question for the first image")
        precondition(drafts[second] == "Question for a different image")
    }

    static func checkRemovalKeepsOtherDrafts() {
        let first = UUID(), second = UUID()
        var drafts = QuestionDrafts()
        drafts[first] = "First"
        drafts[second] = "Second"
        drafts.remove(first)
        precondition(drafts[first].isEmpty)
        precondition(drafts[second] == "Second")
    }
}
