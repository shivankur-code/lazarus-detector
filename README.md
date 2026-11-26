# Lazarus detector

A read-only research detector for **small, dense causal language models**. It asks whether the final feed-forward layer gives a likely next token an unusually large push concentrated in just a few neurons. It needs the checkpoint, tokenizer and eight ordinary sentences: **no secret trigger, known payload, edited-neuron list or clean reference checkpoint**.

This is an independent small-model study inspired by the [Lazarus paper](https://arxiv.org/abs/2611.15118), not the authors' implementation or a validated general defense. **An alarm is not a list of neurons safe to delete. A quiet scan is not a safety certificate.**

## What did the experiments show?

| Question | Observation | Status |
|---|---|---|
| Could the original weight-based detector select a useful repair set? | Across three small bases, 46/60 hand-edited analytic or leak-hardened specimens met the working-backdoor criterion. After silencing the selected neurons, **46/46** produced the exact canary on **0/16** triggered test prompts. Same-size random non-edited sets preserved the behavior in all 46 under the frozen control criterion. | Settings fixed in advance. **0/16** valid rotated specimens were detected or repaired by that method. Clean-prompt drift worsened in 2/46. This is not a general safe repair. |
| How well did the weight test OR a three-step chain detect harder cases? | **22/29** valid specimens detected out of 45 build attempts, below the specified 90% target. Chain alone caught **7/14** valid rotated, split or combined cases, below the 80% target. **0/6** completed fresh clean scans alarmed; one of seven failed to load. | Confirmatory. A controller interruption required resuming unfinished items without changing completed scores. Six clean models are too few to establish a low population false-alarm rate. |
| Did one strong step help? | Adding **peak >= 11** raises detection on those same traces to **27/29**. All five extra alarms are on Qwen3-0.6B: **7/12 to 12/12**. The other bases are unchanged: SmolLM2-135M 9/9; Qwen2.5-0.5B 6/8. | **Exploratory.** The cutoff was chosen after looking. Twelve variants of one Qwen3 base are not twelve independent models. No new positive cohort has verified this gain. |

The repair set came from **weight alignment**, not the top four contributors used for the peak statistic. This repo performs detection only; it never edits or exports weights.

The repair trial used ten phrase/trigger pairs on SmolLM2-135M, Qwen3-0.6B and Qwen2.5-0.5B. A working specimen emitted the exact inert text on at least 12/16 triggered prompts and at most 1/16 matched controls. The same-sized random comparison used three seeds. The harder detector trial used three new pairs across the same bases and five variants. Prompts, starts and specimens sharing a base are not independent replications.

## How the three decisions work

1. **Weight alignment (v4 alarm):** find the vocabulary direction each last-layer neuron resembles most. At least two neurons with sign-corrected cosine similarity >= 0.8 trigger the first alarm. The matching tokens also provide starting points. These are not a dictionary of malicious words.
2. **Domino chain:** try each distinct starting token after ordinary sentences. Measure how much including the final feed-forward output increases a candidate's log probability versus leaving that output out. Estimate how much of the **positive, centered-logit contribution** comes from the top four neurons. For example, a word the model already expects may need little extra push and break a run. Three consecutive likely steps with override >= 3 nats, share >= 0.4 and log probability >= -2 alarm.
3. **Single peak:** a likely step with `override * top_four_share >= 11` alarms even if the next domino doesn't fall. One saved Qwen3 case missed by the chain had `16.67 * 0.896 = 14.94`. This measures concentrated influence on one prediction; it does not prove four malicious neurons caused a backdoor.

The scan takes medians over the eight contexts, grows each path for at most five transitions, considers the eight largest overrides among eligible non-repeated next tokens (log probability >= -3), and retains the best 300 paths by maximum step product. It scores the rounded step fields used in the original runs. `baseline` is v4 OR chain; `combined` also includes peak. We omit payload readout and ordering; they do not affect the v4 alarm.

## Reproduce the counts without a GPU

With Python 3.10 or newer, from this directory:

```text
python replay.py
python -m unittest -v test_detector
```

[Repair evidence](evidence/repair.json) contains anonymous specimen/pair IDs, validity counts, repaired and random-control outcomes, and before/after clean drift. The rotated result is an archived aggregate, not a per-specimen replay. [Detection evidence](evidence/detection.json) contains the original v4 decision and retained `[override, top_four_share, log_probability]` steps grouped by start, plus failed/invalid records. Each file includes the SHA-256 of its source artifact for comparison with the research archive. These are **empirical records, not mathematical proofs**. Replay checks the decision logic and the tables, not the original checkpoint measurements. No model weights, generated commands, triggers, private logs or rental scripts are included.

## Run on a local model

Install [PyTorch](https://pytorch.org/get-started/locally/) for your device and the dependencies in `requirements.txt`. Supply an existing local Hugging Face checkpoint directory with tokenizer and FP32-convertible safetensors:

```text
python -m pip install -r requirements.txt
python detector.py "PATH_TO_CHECKPOINT"
python detector.py "PATH_TO_CHECKPOINT" --device cuda
```

The default is CPU, one compute thread, two starting tokens per batch. Full scans can be expensive even with small batches; use a suitable GPU for larger models. Only local weights are loaded; remote model code and pickle checkpoints are not enabled. Scans must finish to count. The public scanner supports the standard dense Llama/Qwen2/Qwen3 layout; this is not evidence of accuracy on every checkpoint with that layout. Hardware and library versions can affect borderline decisions.

## How to strengthen the evidence

* **Audit now:** `replay.py` also removes the benefit of each newly caught case's strongest starting path. Only **3 of the 5** new alarms have another start above 11. That is a robustness check on old data, not fresh evidence.
* **Next decisive test:** lock this commit and cutoff before a third party selects previously unscored Qwen3 positives and clean checkpoints. Keep labels hidden until both decisions have been recorded for every model. Report paired baseline-to-peak gains, clean false alarms, missing scans and results by base/payload family. Do not count starts or prompt repetitions as independent samples.
* **Test utility separately:** use held-out ordinary tasks to compare model behavior before/after repair with a matched random control. Do not use a peak alarm as a repair list.

After fixing 11, one new small clean checkpoint (Stories15M) scored 1.79 over 627/627 starts. A saved positive SmolLM2 control stopped at a CPU time limit after 344/1,471 starts; it is **inconclusive**, not a negative. Neither is the missing blinded Qwen3 validation. The numerical records for these two scans are in [transfer.json](evidence/transfer.json).
