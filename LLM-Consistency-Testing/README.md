# LLM Consistency Testing

Does a model change its answer when the same question is worded differently? This is a small Python experiment inspired by [KonTest](https://aclanthology.org/2024.findings-emnlp.596/).

The questions are in `data/questions.csv`. There are 28 cases (10 geography and 6 each for math, natural science, and rule-based reasoning), with three yes/no questions per case. For example, “Is 28 divisible by 4?” and “Can 28 counters be arranged into groups of 4 with none left over?” are two versions of the same question.

## Running it

Use Python 3.10 or newer. Run this from the project folder.

```powershell
python experiment.py
```

You can also open the folder in PyCharm and run `experiment.py`. This uses the sample replies in `data/demo_answers.csv` so you can check that the scoring works.

For a local model, start [Ollama](https://docs.ollama.com/windows) and run these two commands.

```powershell
ollama pull llama3.2:1b
python experiment.py --ollama llama3.2:1b
```

To collect replies through a chat interface, first generate a blank CSV.

```powershell
python experiment.py --template results/manual_answers.csv
```

Use the same model for every prompt and start a fresh chat each time. Copy each full reply into `raw_reply`, save as UTF-8 CSV, and run the following command with the actual model name.

```powershell
python experiment.py --answers results/manual_answers.csv --model-label "Your model name"
```

## Reading the output

Look under `results/` for `details.csv` (the questions and replies) and `summary.json` (scores overall and by topic). Accuracy counts correct replies out of all prompts. Consistency counts how often all three replies agree, among cases where all three can be read as yes or no. If the model says `yes` three times and the answer is `no`, the case is consistent but wrong.
