"""CoRe reranker analyzer: document relevance from captured Q/K attention."""
import torch
from typing import Dict, List, Tuple, Optional
import glob
import math

from mia._profiler import PROF
from mia.run_utils import load_and_merge_qk_cache

class CorerAnalyzer:
    ACCEPTS = "qk"

    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        self.layer_to_heads = layer_to_heads

    def analyze(
        self,
        analyzer_spec: Optional[Dict] = None,
        run_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Analyze document relevance using QK artifacts from two generate() passes."""
        if run_ids is None or len(run_ids) < 2:
            raise ValueError("CorerAnalyzer.analyze: pass run_ids=[doc_run_id, na_run_id].")

        PROF.incr("analyzer.corer.calls")

        if not isinstance(analyzer_spec['query_spec'], list):
            analyzer_spec['query_spec'] = [analyzer_spec['query_spec']]
        if not isinstance(analyzer_spec['na_spec'], list):
            analyzer_spec['na_spec'] = [analyzer_spec['na_spec']]

        doc_run_id = run_ids[-2]
        doc_span, query_start, after_instruct, query_end = tuple(map(list, zip(*analyzer_spec['query_spec'])))
        with PROF.timed("analyzer.corer.score_doc"):
            tok_scores, prefill = self.score_documents(doc_run_id, doc_span, query_start, after_instruct, query_end)

        na_run_id = run_ids[-1]
        _, query_start, after_instruct, query_end = tuple(map(list, zip(*analyzer_spec['na_spec'])))
        with PROF.timed("analyzer.corer.score_na"):
            tok_scores_na, _ = self.score_documents(na_run_id, doc_span, query_start, after_instruct, query_end, prefill)
        del prefill

        with PROF.timed("analyzer.kernel"):
         bs = len(doc_span)
         batch_scores = []
         batch_ranking = []
         for i in range(bs):
            doc_scores = torch.zeros(len(doc_span[i]))
            _i = 0
            for tok_score, tok_score_na in zip(tok_scores[i], tok_scores_na[i]):
                calibrated_score = tok_score - tok_score_na
                threshold = calibrated_score.mean() - 2*calibrated_score.std()
                tok_mask = (calibrated_score>threshold)

                tok_score = tok_score * tok_mask
                tok_score_na = tok_score_na * tok_mask
                doc_scores[_i] = (tok_score - tok_score_na).sum().to('cpu')
                _i += 1

            sorted_results = torch.sort(doc_scores, descending=True)
            batch_scores.append(sorted_results.values.tolist())
            batch_ranking.append(sorted_results.indices.tolist())

        del tok_score, tok_score_na, tok_scores, tok_scores_na, calibrated_score, threshold, tok_mask
        torch.cuda.empty_cache()

        return {
            'scores': batch_scores,
            'ranking': batch_ranking,
        }


    def score_documents(
        self,
        run_id: str,
        doc_span,
        query_start_tok_idx,
        after_instruct, 
        query_end_tok_idx,
        past_prefill: Optional[Dict] = None
    ) -> List[torch.Tensor]:
        cache = load_and_merge_qk_cache(self.hook_dir, run_id)
        config = cache["config"]
        if any("scores" in e for e in cache["qk_cache"].values()):
            raise NotImplementedError(
                "CorerAnalyzer requires QK capture (qk_capture='qk'); attention-score "
                "capture (v0.6.0) is only consumed by AttntrackerAnalyzer.")
        bs = len(next(iter(cache["qk_cache"].values()))['q'])

        first_layer = list(cache["qk_cache"].keys())[0]
        k_all_full = False
        if past_prefill is not None:
            na_k = cache["qk_cache"][first_layer]['k_all'][0].shape[0]
            na_q = cache["qk_cache"][first_layer]['q'][0].shape[0]
            k_all_full = na_k > na_q

        if past_prefill is not None and not k_all_full:
            qk_cache = {}
            masks = [~torch.isnan(cache["qk_cache"][list(cache["qk_cache"].keys())[0]]['q'][i]).any(dim=1) for i in range(bs)]
            for module_name in past_prefill.keys():
                merged_q = []
                merged_k = []
                for prefill_q, prefill_k, cache_q, cache_k, mask in zip(
                    past_prefill[module_name]['q'],
                    past_prefill[module_name]['k_all'],
                    cache["qk_cache"][module_name]['q'],
                    cache["qk_cache"][module_name]['k_all'],
                    masks,
                ):
                    valid_q = cache_q[mask]
                    merged_q.append(torch.cat([prefill_q, valid_q], dim=0))
                    merged_k.append(torch.cat([prefill_k, cache_k[mask]], dim=0))
                qk_cache[module_name] = {
                    'q': merged_q,
                    'k_all': merged_k,
                    'layer_num': past_prefill[module_name]['layer_num']
                }
        else:
            qk_cache = cache["qk_cache"]
        prefill_qk_cache = {}

        all_layer = []
        all_key_cache = []
        all_query_cache = []

        for module_name, qk_data in qk_cache.items():
            layer_num = qk_data['layer_num']

            k_all = qk_data['k_all']

            if past_prefill is not None and not k_all_full:
                q_query = [qk_data['q'][i][:query_end_tok_idx[i]-query_start_tok_idx[i]+1, :] for i in range(bs)]
            else:
                q_query = []
                for i in range(bs):
                    q_len = qk_data['q'][i].shape[0]
                    k_len = k_all[i].shape[0]
                    offset = k_len - q_len
                    qs = max(0, query_start_tok_idx[i] - offset)
                    qe = max(0, query_end_tok_idx[i] - offset) + 1
                    q_query.append(qk_data['q'][i][qs:qe, :])

            if past_prefill is None:
                prefill_qk_cache[module_name] = {
                    'q': [qk_data['q'][i][query_start_tok_idx[i]:after_instruct[i]+1, :] for i in range(bs)],
                    'k_all': [qk_data['k_all'][i][:after_instruct[i]+1, :] for i in range(bs)],
                    'layer_num': layer_num
                }

            k_heads, q_heads = [], []
            for i in range(bs):
                query_len = q_query[i].shape[0]
                seq_len = k_all[i].shape[0]

                q_head = q_query[i].view(query_len, config["num_attention_heads"], config["head_dim"])
                q_head = q_head.permute(1, 0, 2)
                q_heads.append(q_head)

                k_head = k_all[i].view(seq_len, config["num_key_value_heads"], config["head_dim"])
                k_head = k_head.permute(1, 0, 2)
                k_heads.append(k_head)

            all_layer.append(layer_num)
            all_query_cache.append(q_heads)
            all_key_cache.append(k_heads)

        batch_doc_results = []
        for j in range(bs):
            if self.layer_to_heads is not None:
                attn_weights = []
                for i in range(len(all_key_cache)):
                    attn_weights.append((self.get_attn_all(all_key_cache[i][j], all_query_cache[i][j])).mean(-2))
                attn_weights = torch.stack(attn_weights)
                attn_weights = attn_weights.sum(0).sum(0)

            else:
                selected_key_cache = [inner[j] for inner in all_key_cache]
                j_all_key_cache = torch.stack(selected_key_cache).squeeze(1)
                selected_query_cache = [inner[j] for inner in all_query_cache]
                j_all_query_cache = torch.stack(selected_query_cache).squeeze(1)
                attn_weights = self.get_attn_head(all_layer, j_all_key_cache, j_all_query_cache)
                attn_weights = attn_weights.mean(-2).sum(0)

            per_doc_results = [None for _ in range(len(doc_span[j]))]
            for i, span in enumerate(doc_span[j]):
                per_doc_results[i] = attn_weights[span[0]:span[1]+1].to('cpu')
            batch_doc_results.append(per_doc_results)

        del attn_weights, all_key_cache, all_query_cache
        torch.cuda.empty_cache()

        return batch_doc_results, prefill_qk_cache

    def get_attn_all(self, key_states, query_states):
        num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(0)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = key_states.unsqueeze(1).expand(num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_heads, kv_seq_len, head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)

        del key_states, query_states
        torch.cuda.empty_cache()

        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2))
        causal_mask = torch.triu(causal_mask, diagonal=-(kv_seq_len-q_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        attn_weights += causal_mask
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        del causal_mask, attn_lses
        torch.cuda.empty_cache()

        return attn_weights

    def get_attn_head(self, all_layer, key_states, query_states):
        num_layers, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = key_states.unsqueeze(2).expand(num_layers, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_layers, num_heads, kv_seq_len, head_dim)

        layer_idx_to_position = {layer: i for i, layer in enumerate(self.layer_to_heads.keys())}
        key_states = torch.cat([
            key_states[layer_idx_to_position[layer_idx], head_indices]
            for layer_idx, head_indices in self.layer_to_heads.items()
        ], dim=0)

        query_states = torch.cat([
            query_states[layer_idx_to_position[layer_idx], head_indices]
            for layer_idx, head_indices in self.layer_to_heads.items()
        ], dim=0)
        torch.cuda.empty_cache()

        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)
        del key_states, query_states
        torch.cuda.empty_cache()

        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2))
        causal_mask = torch.triu(causal_mask, diagonal=-(kv_seq_len-q_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        attn_weights += causal_mask
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        del causal_mask, attn_lses
        torch.cuda.empty_cache()

        return attn_weights
