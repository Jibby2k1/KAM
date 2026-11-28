# Can matched continued training recover sparse attention quality?

All six matched checkpoints finished training. HPG canceled the frozen-holdout evaluation array before it produced a complete paired panel. The results below cover training fidelity, compute cost and checkpoint identity.

Three frozen Llama checkpoints gave lower perplexity and better 4K retrieval under Flash than under all-layer protected shortlisting. Continue the same four live-token adapters under each runtime with the Llama backbone fixed. Cross the training runtime with the evaluation runtime to distinguish sparse-specific recovery from ordinary domain adaptation.

## Fixed pilot and experiment

- Verify that the differentiable sparse training path uses the original Triton routes. Check BF16 outputs against the deployed kernel and gradients against independent masked Torch attention. Treat the discrete selection as fixed during backpropagation. [PyTorch FlexAttention implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/attention/flex_attention.py)
- Select 96 books by hashed ID from the official PG-19 training split. Record object hashes, book IDs and token windows. Each seed uses a fixed 128-window, 4,096-token exposure schedule, matched across arms. Test on the frozen PG-19 and natural retrieval panels only after training. [PG-19 split](https://github.com/google-deepmind/pg19)
- Start each pair from the same parameter and optimizer initialization. Use 128 AdamW steps, peak learning rate \(8\times10^{-5}\), cross-entropy on the final 256 targets per window, and a 0.5 forward-KL penalty toward the matched starting Flash model at sixteen selected targets. Cache the teacher states once per seed. Run three training updates in each arm as a memory, gradient and compiler gate before spending the full training budget.
- Evaluate the fixed last checkpoint, without selection on the test panels, under both Flash and protected shortlisting. Report perplexity and mirrored retrieval at 4K and 8K, including the train-by-evaluation interaction. Keep the same A/B scoring, frozen fixtures and source overlaps as in the prior evaluations.

## Training fidelity and cost

The training sample yielded 277 eligible windows from 96 official training books. Every paired seed consumed the same 128 windows and 524,288 input tokens. The teacher cache held sixteen 4,096-dimensional BF16 hidden states per step, with matching exposure and initial adapter hashes.

On the GPU reference test, the largest absolute forward difference between the serving and differentiable sparse kernels was 0.00390625 at 1K and 4K. Relative gradient errors at 1K versus the independent fixed-mask Torch computation were 0.274% for queries, 0.266% for keys, and 0.252% for values. The same 32-by-64 routing mask was applied in 128-token FlexAttention scheduling tiles. A smaller internal tile and one pipeline stage fit the RTX PRO 6000 compiler's shared-memory limits.

Across three seeds, the measured training loop used 9,068–9,079 input tokens per second for Flash and 7,387–7,615 for the differentiable protected route. Peak allocated memory was about 31.24 GiB in both cases. These measurements cover the 128-update adapter continuation and use an explicit differentiable adapter path. Evaluation uses the frozen serving kernels.

[Paired training receipts and archived checks](../../results/general_benchmarks_sparse_adaptation_training/summary.json) · [Scheduler records](../../results/general_benchmarks_sparse_adaptation_training/scheduler.tsv)
