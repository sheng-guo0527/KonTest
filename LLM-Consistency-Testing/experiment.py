"""Measure answer consistency across three equivalent questions per fact."""

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
INSTRUCTION = "Answer with exactly one word: Yes or No."
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
QUESTION_STYLES = ("direct", "reframed", "scenario")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_questions():
    questions = []
    for case in read_csv(ROOT / "data" / "questions.csv"):
        for variant, style in enumerate(QUESTION_STYLES, start=1):
            question = case[f"question_{variant}"]
            questions.append({
                "case_id": case["case_id"],
                "question_id": f"{case['case_id']}-{variant}",
                "topic": case.get("topic") or "uncategorized",
                "style": style if case.get("topic") else "unspecified",
                "prompt": f"{INSTRUCTION} {question}",
                "expected": case["expected"],
            })
    return questions


def load_replies(path, questions):
    replies = {}
    for row in read_csv(path):
        question_id = row["question_id"]
        if question_id in replies:
            raise ValueError(f"Duplicate reply for {question_id}")
        replies[question_id] = row["raw_reply"]
    for question in questions:
        question_id = question["question_id"]
        if not (replies.get(question_id) or "").strip():
            raise ValueError(f"Missing reply for {question_id} in {path}")
    return replies


def ask_ollama(model, prompt):
    # Send a single user message each time to keep question variants independent.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)["message"]["content"]


def normalize(reply):
    match = re.fullmatch(r"(yes|no)[.!?]?", reply.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else ""


def evaluate(questions, replies):
    rows = []
    groups = {}
    for question in questions:
        raw = replies[question["question_id"]]
        parsed = normalize(raw)
        row = {**question, "raw_reply": raw, "parsed": parsed,
               "correct": parsed == question["expected"]}
        rows.append(row)
        groups.setdefault(question["case_id"], []).append(row)

    groups_by_topic = {}
    for group in groups.values():
        labels = {row["parsed"] for row in group}
        if "" in labels:
            status = "unscorable"
        elif len(labels) == 1:
            status = "consistent"
        else:
            status = "inconsistent"
        for row in group:
            row["group_status"] = status
        groups_by_topic.setdefault(group[0]["topic"], []).append(group)

    metrics = summarize(groups.values())
    metrics["by_topic"] = {
        topic: summarize(topic_groups) for topic, topic_groups in groups_by_topic.items()
    }
    return rows, metrics


def summarize(groups):
    rows = [row for group in groups for row in group]
    counts = [group[0]["group_status"] for group in groups]
    correct = sum(row["correct"] for row in rows)
    parsed_count = sum(bool(row["parsed"]) for row in rows)
    consistent = counts.count("consistent")
    scorable = consistent + counts.count("inconsistent")
    return {
        "total_answers": len(rows),
        "correct_answers": correct,
        "answer_accuracy": correct / len(rows),
        "unparseable_answers": len(rows) - parsed_count,
        "total_groups": len(groups),
        "scorable_groups": scorable,
        "consistent_groups": consistent,
        "inconsistent_groups": counts.count("inconsistent"),
        "unscorable_groups": counts.count("unscorable"),
        "consistent_but_wrong_groups": sum(
            group[0]["group_status"] == "consistent" and not group[0]["correct"]
            for group in groups
        ),
        "group_consistency": consistent / scorable if scorable else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--template", type=Path, help="Export a blank answer CSV")
    mode.add_argument("--answers", type=Path, help="Score manually collected replies")
    mode.add_argument("--ollama", metavar="MODEL", help="Collect local model replies")
    parser.add_argument("--model-label", help="Model name for --answers")
    args = parser.parse_args()
    if args.answers and not args.model_label:
        parser.error("--answers requires --model-label")

    questions = load_questions()
    if args.template:
        if args.template.exists():
            parser.error(f"File already exists: {args.template}")
        write_csv(args.template, ["question_id", "prompt", "raw_reply"], [
            {"question_id": q["question_id"], "prompt": q["prompt"]} for q in questions
        ])
        print(f"Answer template: {args.template}")
        return

    try:
        if args.ollama:
            source, model = "ollama", args.ollama
            replies = {}
            for number, question in enumerate(questions, start=1):
                print(f"Collecting {number}/{len(questions)}", flush=True)
                replies[question["question_id"]] = ask_ollama(model, question["prompt"])
        elif args.answers:
            source, model = "manual", args.model_label
            replies = load_replies(args.answers, questions)
        else:
            source, model = "demo", None
            replies = load_replies(ROOT / "data" / "demo_answers.csv", questions)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Could not collect replies: {error}\n")

    rows, metrics = evaluate(questions, replies)
    summary = {"source": source, "model": model, **metrics}
    directory = ROOT / "results" / source
    write_csv(directory / "details.csv", list(rows[0]), rows)
    report = json.dumps(summary, indent=2)
    (directory / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved results in {directory}")


if __name__ == "__main__":
    main()
