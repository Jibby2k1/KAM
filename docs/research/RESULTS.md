# KAM research results

## Multi-hop evidence selection

The [held-out HotpotQA supporting-link study](../../reports/nonlinear_retrieval_natural/SUPPORT_LINKS.md) compares a question-conditioned pair scorer with one-pass and sequential retrieval on shared dense pools. It reports linked and unlinked gold evidence separately and tests the ranking of supporting links against distractor links directly.

## Corpus retrieval and serving

At matched isolated latency on two independent HotpotQA cohorts (512 questions total), exact dense retrieval put both gold documents in the candidate pool for 349 questions, compared with 302 for hybrid and 266 for BM25. Joint exact match was 85, 89, and 91 respectively. Paired intervals covered zero. The change in rank from pool coverage to downstream joint exact match identifies pair selection as a separate bottleneck. [L4 aggregate](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/exact_threads_l4/summary.json) · [Independent replication](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/exact_replicate_l4/summary.json)

On an RTX GPU, FP32 lexical pruning in the hybrid pipeline reduced p95 latency from 240–253 ms to 78–80 ms across two traces offered at about 32 requests per second. Joint exact match was 53 versus 52 for the exact pipeline on the 384-question replication. Fifteen answers changed and the paired interval covered zero. [Replication results](../../results/nonlinear_retrieval_natural/fullwiki_native_v1/hybrid_replicate_v1/summary.json)

## Revision memory

A frozen conservative gate rejected all 576 prompts for 96 unseen registry identities while answering 48 of 192 supported-alias prompts. Supplying a verified identity raised supported coverage to 192 of 192. These figures measure coverage and abstention on that fixture. [Registry holdout results](../../results/nonlinear_retrieval_natural/release_registry_sample_v1/summary.json)

## Model and controlled experiments

See the [architecture specification](../../Kernel_Adaptive_Memory_Research_Specification.pdf), the [core experiment guide](../../EXPERIMENTS.md), and the [Phase IV–VI documentation index](../../LLM_NAVIGATION.md) for controlled KAM model experiments. Natural-task experiments evaluate the transfer of routing ideas using separate retrieval components.
