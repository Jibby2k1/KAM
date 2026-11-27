# Longer-context KAM and dot-attention benchmark

This experiment compares two new roughly 10M-parameter byte language models on the same five test sets as the [128-byte TinyStories study](GENERAL_BENCHMARKS_PROTOCOL.md). Both use 2,048-byte training windows and the same sampled inputs, optimizer, learning rate schedule and training budget. T-KAM-F uses fixed support keys and learns the value vectors in each memory module. T-WIDE spends its matched total parameter budget on an ordinary dot-attention decoder with a wider feed-forward network. The KAM arm uses top-four lookup over 1,024 supports per layer. The earlier TinyStories experiment used 4,096 supports per layer.

## HPG results

| Benchmark | KAM | Dot | KAM − dot (pp) | Paired 95% item interval | Full initial prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 0.00% (0/164) | 0.00% (0/164) | +0.00 | No discordant items | 164/164 |
| MBPP | 0.00% (0/500) | 0.00% (0/500) | +0.00 | No discordant items | 500/500 |
| GPQA Diamond | 23.74% (47/198) | 26.26% (52/198) | -2.53 | [-6.57, +1.52] | 197/198 |
| MMLU | 23.09% (3,242/14,042) | 22.92% (3,218/14,042) | +0.17 | [-0.14, +0.48] | 13,840/14,042 |
| GSM8K | 0.23% (3/1,319) | 0.68% (9/1,319) | -0.45 | [-0.91, -0.08] | 1,319/1,319 |

- HumanEval grading outcomes. KAM had 164 failed tests. Dot had 164 failed tests.
- MBPP grading outcomes. KAM had 481 no definition, 19 syntax error. Dot had 493 no definition, 7 syntax error.
- GSM8K produced a parseable marked numeric answer on 234/1,319 prompts for KAM and 683/1,319 for dot.

| Training measure | KAM | Dot |
| --- | ---: | ---: |
| Total parameters | 9,993,170 | 10,002,800 |
| Test byte perplexity, mixture | 3.686 | 3.681 |
| Test NLL, wiki (nats/byte) | 1.7220 | 1.7195 |
| Test NLL, code (nats/byte) | 1.4854 | 1.4844 |
| Test NLL, math (nats/byte) | 0.2890 | 0.2897 |
| Elapsed training and scheduled validation (s) | 260.8 | 191.8 |
| Peak allocated GPU memory (GiB) | 6.07 | 6.05 |

The GPQA and MMLU paired intervals included zero. The models chose A on 183 of 198 GPQA items each, and on 97.7% and 95.2% of MMLU items for KAM and dot respectively. The dot arm produced more parseable GSM8K answers and had six more exact matches. The item-level arithmetic difference reflects this single checkpoint pair and includes output-format behavior. Both coding scores remained at the floor, despite improved syntax on HumanEval.

[Paired results](../../results/general_benchmarks_long/summary.json) · [Per-item outcomes](../../results/general_benchmarks_long/paired-outcomes.json.gz) · [Packed corpus manifest](../../results/general_benchmarks_long/corpus.json)

## Data and training

Each batch draws four windows from WikiText-103 prose, two from CodeSearchNet Python functions, and two from independently generated arithmetic word problems. The arithmetic examples teach a numeric answer format and simple operations, including combinations of multiplication and addition or subtraction. They use four fixed templates and separate seeds by split. WikiText-103 uses the first raw train shard and its released validation and test splits. CodeSearchNet uses the released Python train, validation and test splits. The preparation script takes bounded document prefixes of each source split. It excludes exact normalized duplicate documents across splits and strips training documents with exact normalized test prompts of at least eight words as substrings. Short, reformatted, or paraphrased overlap can remain. The benchmark inputs are used only for this exclusion filter. The exact prompt filter found zero matches in the selected training documents.

Both models train from scratch with the same initialization seed, 8,192 AdamW updates and 134,217,728 byte-token predictions. Training uses BF16 autocast, global gradient norm clipping at one, a 256-update warmup to 0.0004 and cosine decay to 0.00004. The data index stream and packed corpus hashes identify the exact exposure. Validation and test NLL are estimated using the same independently seeded windows per arm in each source. Test perplexity is byte-level and evaluated within this mixture.

Sources and revisions are pinned in the [data fetcher](../../scripts/fetch_kam_long_corpus.sh). See [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) and [CodeSearchNet](https://github.com/github/CodeSearchNet).

## Benchmark protocol

Reuse the same pinned test files, zero-shot prompts, per-item GPQA option shuffle, answer-letter scoring, byte generation limits and stopping rules, strict GSM8K number extraction and isolated coding tests as in the [earlier protocol](GENERAL_BENCHMARKS_PROTOCOL.md). Both models take the most recent 2,048 bytes on each forward. Report the number of prompts that fit before generation and match all per-item decisions by benchmark index. The summary records parse success separately from answer correctness.

One pair ran per training recipe. Intervals describe item-level score differences for that pair. Several factors differ from the earlier TinyStories study, including training corpus, budget, support count and context, so the comparison across studies cannot identify the individual effect of context length.

The HPG preparation job was 53034690, the paired B200 training array was 53034691, the L4 benchmark array was 53034692, and the isolated CPU graders were 53037266 for HumanEval and 53037258 for MBPP. The [training manifest](../../results/general_benchmarks_long/manifest.jsonl) records both architecture rows. The [KAM log](../../results/general_benchmarks_long/train-T-KAM-F.json) and [dot log](../../results/general_benchmarks_long/train-T-WIDE.json) record validation history, matching exposure digests, and the separately evaluated test NLL.

## Reproduce on HPG

Prepare the five benchmark assets with the [benchmark fetcher](../../scripts/fetch_kam_general_benchmarks.sh), the training sources with the [corpus fetcher](../../scripts/fetch_kam_long_corpus.sh), and the PyArrow 20.0.0 runtime and Python grading container as described in the [earlier protocol](GENERAL_BENCHMARKS_PROTOCOL.md). Stage the [public model source](https://github.com/Jibby2k1/KAM/tree/814fe90/kam) under the run directory's repo path, and place the following Python and Slurm entry points in the run directory.

1. Submit the [data preparation job](../../slurm/kam_general_long_prepare.sbatch) to run the [corpus builder](../../scripts/prepare_kam_long_corpus.py) and write the paired training manifest.
2. Submit the [matched training array](../../slurm/kam_general_long_train.sbatch) to run the [trainer](../../scripts/train_kam_general_long.py) on two B200 GPUs after preparation. The trainer writes atomic optimizer checkpoints every 1,024 updates and can resume from those checkpoints.
3. Submit the [evaluation array](../../slurm/kam_general_long_eval.sbatch), then the [isolated coding graders](../../slurm/kam_general_code_grading.sbatch). Run the [paired summarizer](../../scripts/summarize_kam_general_benchmarks.py) on the numeric results. Dataset text and code completions remain in cluster storage.
