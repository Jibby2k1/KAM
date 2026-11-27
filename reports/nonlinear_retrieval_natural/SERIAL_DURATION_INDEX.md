# Exact event lookup after swapping durations

An index over the original serial schedule answered the same queries as rebuilding the full timeline after each edit. On an HPG CPU, indexed lookup took 0.68 μs per request for 32 events and 1.15 μs for 16,384. Full reconstruction took 1.76 and 446.97 μs per request.

Let \(E_i\) be the prefix sum through event \(i\), which represents the end of its half-open interval. Swap durations at \(a<b\) and let \(\delta=d_b-d_a\). The updated \(E_i\) gains \(\delta\) only for \(a\le i<b\). The anchor start shifts by the same amount if its preceding event falls in that range. Once the query clock is anchored, comparing it to the two region boundaries identifies the slice to binary-search. Inside the shifted slice, subtract \(\delta\) from the search key. The exact index uses one prefix scan per plan and logarithmic lookup per query.

| Events | HPG rebuild | HPG indexed | HPG indexed 1,024 requests per batch | HPG prefix setup |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.761 μs | 0.677 μs | 0.178 μs/request | 1.333 μs |
| 256 | 9.190 μs | 0.721 μs | 0.254 μs/request | 8.405 μs |
| 2,048 | 61.498 μs | 0.880 μs | 0.386 μs/request | 55.038 μs |
| 16,384 | 446.975 μs | 1.145 μs | 0.488 μs/request | 403.151 μs |

On Darwin arm64, rebuild and indexed costs ranged from 0.48 versus 0.17 μs at 32 events to 143.34 versus 0.32 μs at 16,384. These are medians over eight randomized timing passes with matched 1,024-request numeric workloads. The batched column represents throughput on ready NumPy inputs. Timings exclude parsing event text and packing the query arrays. The one-time prefix setup time is shown separately. Both hosts matched all 3,519 original, sham, and intervention targets, plus 5,760 constructed boundary cases with anchors on both sides of the edit. All timing input fingerprints match. HPG job 52964206 used one bound CPU core. [Timing harness](../../scripts/benchmark_kam_duration_index.py) · [Matched host records](../../results/nonlinear_retrieval_natural/duration_index_v1/)
