from __future__ import annotations

import re

_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've",
    "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're",
    "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}


def normalize_vqa_answer(answer: str) -> str:
    """Apply the official VQA v2 punctuation, number, article, and contraction rules."""
    text = str(answer).replace("\n", " ").replace("\t", " ").strip()
    output = text
    for punctuation in _PUNCTUATION:
        if punctuation + " " in text or " " + punctuation in text or _COMMA_STRIP.search(text):
            output = output.replace(punctuation, "")
        else:
            output = output.replace(punctuation, " ")
    output = _PERIOD_STRIP.sub("", output)
    words = []
    for word in output.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def normalize_vqa_answer_group(answers: list[str]) -> list[str]:
    """Match the official evaluator's rule: normalize only when annotators disagree."""
    cleaned = [str(answer).replace("\n", " ").replace("\t", " ").strip() for answer in answers]
    if len(set(cleaned)) <= 1:
        return cleaned
    return [normalize_vqa_answer(answer) for answer in cleaned]


def vqa_soft_score(answers: list[str], prediction: str) -> float:
    cleaned = [str(answer).replace("\n", " ").replace("\t", " ").strip() for answer in answers]
    predicted = str(prediction).replace("\n", " ").replace("\t", " ").strip()
    if len(set(cleaned)) > 1:
        cleaned = [normalize_vqa_answer(answer) for answer in cleaned]
        predicted = normalize_vqa_answer(predicted)
    if not cleaned:
        return 0.0

    scores = []
    for held_out, _ in enumerate(cleaned):
        matching = sum(
            answer == predicted
            for index, answer in enumerate(cleaned)
            if index != held_out
        )
        scores.append(min(matching / 3.0, 1.0))
    return sum(scores) / len(scores)


def vqa_soft_targets(answers: list[str]) -> dict[str, float]:
    """Return exact leave-one-annotator-out scores for each distinct answer."""
    cleaned = [str(answer).replace("\n", " ").replace("\t", " ").strip() for answer in answers]
    if len(set(cleaned)) > 1:
        cleaned = [normalize_vqa_answer(answer) for answer in cleaned]
    return {answer: vqa_soft_score(answers, answer) for answer in dict.fromkeys(cleaned)}


def official_vqa_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    require_exact_ids: bool = True,
) -> dict:
    """Compute the official VQA v2 leave-one-annotator score and type breakdowns."""
    prediction_ids = [int(row["question_id"]) for row in predictions]
    truth_ids = [int(row["question_id"]) for row in ground_truth]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("Predictions contain duplicate question IDs.")
    if len(set(truth_ids)) != len(truth_ids):
        raise ValueError("Ground truth contains duplicate question IDs.")
    prediction_by_id = {int(row["question_id"]): str(row["answer"]) for row in predictions}
    truth_by_id = {int(row["question_id"]): row for row in ground_truth}
    if require_exact_ids and prediction_by_id.keys() != truth_by_id.keys():
        missing = sorted(truth_by_id.keys() - prediction_by_id.keys())[:5]
        extra = sorted(prediction_by_id.keys() - truth_by_id.keys())[:5]
        raise ValueError(f"Prediction IDs do not match validation IDs; missing={missing}, extra={extra}")
    if not truth_by_id:
        raise ValueError("Cannot evaluate an empty VQA validation set.")

    overall_scores: list[float] = []
    answer_scores: dict[str, list[float]] = {}
    question_scores: dict[str, list[float]] = {}
    for question_id, row in truth_by_id.items():
        if question_id not in prediction_by_id:
            continue
        score = vqa_soft_score(row.get("answers", []), prediction_by_id[question_id])
        overall_scores.append(score)
        answer_scores.setdefault(str(row.get("answer_type", "unknown")), []).append(score)
        question_scores.setdefault(str(row.get("question_type", "unknown")), []).append(score)

    def percent(scores: list[float]) -> float:
        return round(100.0 * sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "overall": percent(overall_scores),
        "per_answer_type": {key: percent(values) for key, values in sorted(answer_scores.items())},
        "per_question_type": {key: percent(values) for key, values in sorted(question_scores.items())},
        "evaluated_questions": len(overall_scores),
    }
