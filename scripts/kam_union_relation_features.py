"""Exact title and coverage features for original and expanded document pools."""

import numpy as np
import torch

from audit_kam_natural_data import words
from run_kam_fullwiki_support_links import STOP


def bitset_relation_features(example, pairs):
    """Return the existing exact title edges and five lexical relation features."""
    question = set(words(example["question"])) - STOP
    scale = max(len(question), 1)
    bit = {token: 1 << position for position, token in enumerate(question)}
    trie, names, bodies, coverage = {}, [], [], []
    for i, (title, text) in enumerate(example["context"]):
        name = words(title)
        body = words(" ".join(text))[:120]
        names.append(set(name))
        bodies.append(body)
        coverage.append(sum(bit.get(token, 0) for token in (set(body) | names[-1])))
        if len(" ".join(name)) >= 3:
            node = trie
            for token in name:
                node = node.setdefault(token, {})
            node.setdefault(None, []).append(i)
    n = len(bodies)
    graph = np.zeros((n, n), dtype=bool)
    local_counts = np.zeros((n, n), dtype=int)
    for i, body in enumerate(bodies):
        for start, token in enumerate(body):
            node = trie.get(token)
            end = start
            while node is not None:
                for j in node.get(None, ()):
                    graph[i, j] = True
                    window = body[max(0, start - 12):start] + body[end + 1:end + 13]
                    hits = len((set(window) & question) - names[j])
                    local_counts[i, j] = max(local_counts[i, j], hits)
                end += 1
                node = node.get(body[end]) if end < len(body) else None
    left, right = pairs
    sizes = [mask.bit_count() for mask in coverage]
    content = torch.tensor([((coverage[i] | coverage[j]).bit_count() / scale,
                             min(sizes[i], sizes[j]) / scale,
                             (coverage[i] & coverage[j]).bit_count() / scale)
                            for i, j in zip(left.tolist(), right.tolist(), strict=True)])
    local = torch.as_tensor(local_counts / scale, dtype=torch.get_default_dtype())
    fwd, rev = local[left, right], local[right, left]
    return torch.from_numpy(graph), torch.cat((content, torch.stack((torch.maximum(fwd, rev), fwd + rev), -1)), -1)

def induced_original_features(union_features, original_size, retained_pairs):
    """Project the union features onto the pairs in the original prefix."""
    graph, lexical = union_features
    return graph[:original_size, :original_size].contiguous(), lexical[retained_pairs]
