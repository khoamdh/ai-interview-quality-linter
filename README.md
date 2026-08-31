# ai-interview-quality-linter

A lightweight Python tool for screening the quality of AI-led
interview conversations.

The linter applies deterministic and heuristic checks to identify
patterns such as:

- repeated or rephrased questions
- leading questions
- assumed emotions
- unproductive pressing loops
- employee signals that are not developed
- forced answer repetition

## Quality outcomes

Each conversation is classified into one of five screening outcomes:

- `INSIGHTFUL_CLEAN`
- `INSIGHTFUL_WITH_DEFECTS`
- `NO_INSIGHT_MISSED_SIGNAL`
- `NO_INSIGHT_AI_LOOP`
- `NO_INSIGHT_NO_SIGNAL`

## How it works

```text
Employee signal
      ↓
Did the interviewer develop it?
      ↓
Was useful insight produced?
      ↓
Were interviewing defects detected?
      ↓
Conversation outcome
```

## Usage

```bash
python src/transcript_linter.py examples/sample_conversations.json \
  --out-csv results.csv \
  --out-json results.json \
  --summary summary.json
