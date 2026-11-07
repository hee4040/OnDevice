"""Actual-checkpoint smoke check without ImageNet; never an accuracy experiment."""
import argparse
import json

import torch

from evaluate_drop import ROOT, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from layerwise_drop import experiment_configs, layerwise_logits
from selective_refinement import forward_narrow_with_attention, make_update_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(args.repo.resolve(), args.checkpoint.resolve())
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  input='2 seeded random tensors; not ImageNet; no accuracy claim', checks=[])
    with torch.inference_mode():
        images = torch.randn(2, 3, 224, 224, device='cuda')
        features, _, scores = forward_narrow_with_attention(model, images)
        x0 = wide_input(model, images, features)
        full = wide_logits(model, x0)
        official = model.forward_head(model.forward_features(images, 1., features, (3, 6)))
        torch.testing.assert_close(full, official, rtol=1e-5, atol=1e-6)
        for ratio in (.25, .5, .75, 1.):
            reference = wide_logits(model, x0, make_update_mask(scores, ratio))
            actual = layerwise_logits(model, x0, scores, [ratio] * 12)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            result['checks'].append(dict(uniform_ratio=ratio, max_abs_error=(actual - reference).abs().max().item()))
        for config in experiment_configs():
            observed = []
            handles = [block.register_forward_pre_hook(lambda module, inputs: observed.append(inputs[0].shape[1]))
                       for block in model.blocks]
            try:
                actual = layerwise_logits(model, x0, scores, config['ratios'])
            finally:
                for handle in handles:
                    handle.remove()
            assert observed == config['counts'], (config['name'], observed)
            assert actual.shape == (2, 1000) and bool(torch.isfinite(actual).all())
            result['checks'].append(dict(name=config['name'], block_token_counts=observed, finite_logits=True))
    output = ROOT / 'results/layerwise-model-check.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
