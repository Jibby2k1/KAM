"""Rowwise INT8 indexing and padded integer matrix products for corpus search."""

import torch

INT_QUERY_ROWS = 32
BLOCK = 65536


def int8_encode(vectors):
    length, dim = vectors.shape
    assert dim % 32 == 0
    padded = ((length + 15) // 16) * 16
    storage = torch.zeros((padded, dim), dtype=torch.int8, device=vectors.device)
    scales = torch.zeros(padded, dtype=torch.float32, device=vectors.device)
    for i in range(0, length, BLOCK):
        rows = vectors[i:i+BLOCK]
        norm = rows.abs().amax(-1).clamp_min(1e-12)
        storage[i:i+len(rows)] = (rows * (127/norm[:, None])).round().clamp(-127, 127).to(torch.int8)
        scales[i:i+len(rows)] = norm / 127
    return storage, scales


def int8_query(queries):
    padded = torch.zeros((INT_QUERY_ROWS, queries.shape[-1]), dtype=torch.int8, device=queries.device)
    norm = queries.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    padded[:len(queries)] = (queries * (127/norm)).round().clamp(-127, 127).to(torch.int8)
    return padded


def integer_product(queries, bank, scales):
    # Query scale is positive and constant across candidates in a row.
    integer = torch._int_mm(int8_query(queries), bank.T)
    return integer[:len(queries)].float().mul_(scales)
