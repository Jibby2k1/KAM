"""Share frozen decoder activations between a teacher and its adapted copy."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch


@contextmanager
def reuse_prefix(model, stack, last):
    """Replay the teacher's frozen prefix during a paired student forward."""
    captured = [None] * (last + 1)

    def reuse(index, forward):
        def run(*args, **kwargs):
            if not stack.active:
                output = forward(*args, **kwargs)
                if isinstance(output, tuple):
                    assert len(output) == 1
                    captured[index] = output if index == last else (None,)
                else:
                    assert isinstance(output, torch.Tensor)
                    captured[index] = output if index == last else None
                return output
            if index == last:
                output = captured[index]
                assert output is not None
                captured[index] = None
                return output
            hidden = args[0] if args else kwargs["hidden_states"]
            return (hidden,) if isinstance(captured[index], tuple) else hidden

        return run

    with ExitStack() as contexts:
        for index, layer in enumerate(model.model.layers[:last + 1]):
            contexts.enter_context(patch.object(layer, "forward", reuse(index, layer.forward)))
        yield
