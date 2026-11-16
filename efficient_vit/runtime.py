"""GPU graph stages with an explicit host gate; returned logits remain valid."""
import torch
from .policy import PRESETS


class GraphStage:
    def __init__(self, fn, examples):
        self.inputs = tuple(x.clone() for x in examples)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = fn(*self.inputs)

    def __call__(self, *inputs):
        for destination, source in zip(self.inputs, inputs):
            if destination.shape != source.shape or destination.dtype != source.dtype or destination.device != source.device:
                raise ValueError("Graph input must match captured shape, device and dtype")
            destination.copy_(source)
        self.graph.replay()
        return self.outputs


class Engine:
    def __init__(self, model, runtime="eager"):
        if runtime not in ("eager", "graph"):
            raise ValueError("Unknown runtime")
        if runtime != "eager" and next(model.parameters()).device.type != "cuda":
            raise ValueError("Accelerated runtimes require CUDA")
        self.model, self.runtime = model, runtime
        self.narrow_call = lambda images: model.narrow(images, return_embedding=True)
        self.wide_calls = {r: (lambda base, f, a, ratio=r: model.refine(base, f, a, ratio)) for r in (.25, .5, .75, 1.)}
        if runtime == "graph":
            with torch.inference_mode():
                parameter = next(model.parameters())
                x = torch.zeros(1, 3, 224, 224, device=parameter.device, dtype=parameter.dtype)
                features, _, attention, _, embedding = model.narrow(x, return_embedding=True)
                self.narrow_call = GraphStage(self.narrow_call, (x,))
                self.wide_calls = {r: GraphStage(fn, (embedding, features, attention)) for r, fn in self.wide_calls.items()}

    @torch.inference_mode()
    def predict(self, images, policy="balanced", narrow_result=None):
        policy = PRESETS[policy] if isinstance(policy, str) else policy
        if self.runtime == "graph" and len(images) != 1:
            raise ValueError("Graph runtime captures batch 1; use eager for batched evaluation")
        features, logits, attention, entropy, embedding = self.narrow_call(images) if narrow_result is None else narrow_result
        if len(images) == 1:
            # One scalar read, no shape discovery or hooks inside captured stages.
            e = entropy.item()
            code = -1 if e < policy.gate else sum(e >= boundary for boundary in policy.boundaries)
            if code >= 0:
                logits = self.wide_calls[policy.ratios[code]](embedding, features, attention)
            return logits.clone(), torch.tensor([code], device=images.device), entropy.clone()
        codes = policy.codes(entropy)
        output = logits.clone()
        for code, ratio in enumerate(policy.ratios):
            selected = (codes == code).nonzero(as_tuple=True)[0]
            if len(selected):
                wide = self.wide_calls[ratio](embedding[selected], features[selected], attention[selected])
                output.index_copy_(0, selected, wide)
        return output, codes, entropy
