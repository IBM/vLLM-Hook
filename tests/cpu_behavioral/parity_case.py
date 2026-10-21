"""Run one real MIA scenario and save observations on the current machine."""
import argparse
from importlib.metadata import version
import json
import multiprocessing as mp
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True,
                        choices=('hidden_states', 'attention_tracker', 'core_reranker', 'activation_steer'))
    parser.add_argument('--backend', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from vllm import SamplingParams
    from mia import MiaLLM, register_plugins
    from backend_support import assert_backend
    from run_core_reranker_cpu import _prompt_and_spec

    mp.set_start_method('spawn', force=True)
    assert_backend()
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_plugins()
    manifest = json.loads(args.manifest.read_text())
    model = manifest['model_path']
    metadata = dict(manifest, scenario=args.scenario, dtype='float32', seed=0,
                    eager=True, max_tokens=1, temperature=0.0, allow_tf32=False)
    folder = args.output.parent / (args.backend + '-' + args.scenario + '-hooks')
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / 'config.json'
    if args.scenario == 'hidden_states':
        worker, analyzer = 'capture_hs', 'hidden_states'
        settings = {'hidden_states': {'layers': [1, 2], 'mode': 'last_token'}}
    elif args.scenario == 'activation_steer':
        worker, analyzer = 'steer', None
        width = json.loads((Path(model) / 'config.json').read_text())['hidden_size']
        vector = torch.sin(torch.arange(width, dtype=torch.float32))
        vector_file = folder / 'steering.pt'
        torch.save({'dir': vector}, vector_file)
        metadata['steering'] = {'vector': 'sin(arange(hidden_size))', 'width': width,
                                'layer': 1, 'coefficients': [0.0, 1.0]}
        settings = {'steering': {'method': 'add_vector', 'coefficient': 1.0,
                                'optimal_layer': 1, 'vector_path': str(vector_file),
                                'apply_at_all_positions': True}}
    else:
        worker = 'capture_qk'
        analyzer = 'attn_tracker' if args.scenario == 'attention_tracker' else 'core_reranker'
        settings = {'params': {'important_heads': [[0, 0]]},
                    'hookq': {'hookq_mode': 'last_token' if args.scenario == 'attention_tracker'
                              else 'all_tokens'}}
    # Paths differ by backend; logical settings must match exactly.
    metadata['config'] = json.loads(json.dumps(settings))
    if args.scenario == 'activation_steer':
        metadata['config']['steering'].pop('vector_path')
    config.write_text(json.dumps(settings))
    llm = MiaLLM(model=model, worker_name=worker, analyzer_name=analyzer,
                 config_file=str(config), hook_dir=str(folder), dtype=torch.float32,
                 seed=0, max_model_len=128, gpu_memory_utilization=0.25,
                 enable_hook=True, enable_prefix_caching=False, enforce_eager=True)

    def plain(value):
        if isinstance(value, torch.Tensor):
            return {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'values': value.detach().cpu().float().tolist()}
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if hasattr(value, 'tolist'):
            return plain(value.tolist())
        return value

    def generate(prompts, **kwargs):
        params = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
        return llm.generate(prompts, params, **kwargs)

    try:
        if args.scenario == 'hidden_states':
            prompts = ['Hidden states test.']
            outputs = generate(prompts, use_hook=True)
            stats = llm.analyze(analyzer_spec={'reduce': 'none'}, probes=outputs[0].probes)
            values = {'hidden_states': plain(stats['hidden_states'])}
            if len(stats['hidden_states']) != 2:
                raise RuntimeError('Expected two captured hidden-state layers')
            width = llm.llm.llm_engine.model_config.hf_config.hidden_size
            for tensors in stats['hidden_states'].values():
                if len(tensors) != len(prompts) or any(t.shape != (width,) for t in tensors):
                    raise RuntimeError('Unexpected hidden-state shape or prompt count')
        elif args.scenario == 'attention_tracker':
            prompts = ['Instruction: answer yes. Data: this is a parity test.',
                       'Instruction: answer no. Data: this is another parity test.']
            outputs = generate(prompts, use_hook=True)
            ranges = []
            for prompt in prompts:
                length = len(llm.tokenizer(prompt)['input_ids'])
                ranges.append([(0, max(1, length // 2)), (max(1, length // 2), length)])
            metadata['ranges'] = ranges
            stats = llm.analyze(analyzer_spec={'input_range': ranges, 'attn_func': 'sum_normalize'},
                                probes=outputs[0].probes)
            values = {'scores': plain(stats['score'])}
            if len(stats['score']) != len(prompts):
                raise RuntimeError('Missing attention scores')
        elif args.scenario == 'core_reranker':
            prompt, query, count = _prompt_and_spec(llm.tokenizer, 'Which city is older: Rome or New York?')
            baseline, na, _ = _prompt_and_spec(llm.tokenizer, 'N/A')
            prompts = [prompt, baseline]
            metadata.update(query_spec=query, na_spec=na)
            generate(prompt, save_to_disk=True, run_id='parity-doc')
            generate(baseline, save_to_disk=True, run_id='parity-na')
            stats = llm.analyze(analyzer_spec={'query_spec': query, 'na_spec': na},
                                run_ids=['parity-doc', 'parity-na'])
            values = {'scores': plain(stats['scores']), 'ranking': plain(stats['ranking'])}
            if len(stats['scores'][0]) != count or len(stats['ranking'][0]) != count:
                raise RuntimeError('Missing reranker scores/ranking')
        else:
            prompts = ['The capital of France is']
            # Fixed token IDs avoid comparing different top-k sets across backends.
            ids = [4, 10, 42, 100, 200, 500, 1000, 2000, 5000, 10000]
            metadata['logprob_token_ids'] = ids
            observed = []
            for coefficient in (0.0, 1.0):
                params = SamplingParams(temperature=0.0, max_tokens=1, seed=0,
                                        logprobs=len(ids), logprob_token_ids=ids,
                                        extra_args={'steer': {'coefficient': coefficient}})
                outputs = llm.generate(prompts, params, use_hook=True)
                scores = outputs[0].outputs[0].logprobs[0]
                observed.append([float(scores[token].logprob) for token in ids])
            effect = [b - a for a, b in zip(*observed)]
            if max(abs(v) for v in effect) <= 1e-5:
                raise RuntimeError('Nonzero steering produced no measured effect')
            values = {'baseline_logprobs': observed[0], 'steered_logprobs': observed[1],
                      'effect': effect}
        metadata['prompts'] = prompts
        metadata['input_token_ids'] = [llm.tokenizer(p)['input_ids'] for p in prompts]
        # Validate before writing; allow_nan=False rejects non-finite observations.
        result = {'backend': args.backend, 'metadata': metadata, 'values': values,
                  'versions': {name: version(name) for name in ('torch', 'vllm', 'triton', 'vllm-hook-mia')}}
        encoded = json.dumps(result, allow_nan=False)
    finally:
        llm.llm_engine.engine_core.shutdown()
    args.output.write_text(encoded)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
