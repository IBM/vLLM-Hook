# tests/controls/test_corer.py
import pytest
import torch
from typing import List

from vllm_hook_plugins import HookLLM, register_plugins
from tests.conftest import ensure_config_for_model


TEST_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

def apply_chat_template_and_get_ranges(tokenizer, model_name: str, query: str, documents: List[str]):
    # setup prompts
    off_set = 0
    if 'granite' in model_name.lower():
        prompt_prefix = '<|start_of_role|>user<|end_of_role|>'
        prompt_suffix = '<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
    elif 'llama' in model_name.lower():
        prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
        prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    elif 'mistral' in model_name.lower():
        prompt_prefix = '[INST]'
        prompt_suffix = '[/INST]'
        off_set = 1
    elif 'phi' in model_name.lower():
        prompt_prefix = '<|im_start|>user<|im_sep|>'
        prompt_suffix = '<|im_end|><|im_start|>assistant<|im_sep|>'
    retrieval_instruction = ' Here are some paragraphs:\n\n'
    retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: '
    
    doc_span = []
    query_start_idx = None
    query_end_idx = None

    llm_prompt = prompt_prefix + retrieval_instruction

    for i, doc in enumerate(documents):

        llm_prompt += f'[document {i+1}]'
        start_len = len(tokenizer(llm_prompt).input_ids)

        llm_prompt += ' ' + " ".join(doc)
        end_len = len(tokenizer(llm_prompt).input_ids) - off_set

        doc_span.append((start_len, end_len))
        llm_prompt += '\n\n'

    start_len = len(tokenizer(llm_prompt).input_ids)

    llm_prompt += retrieval_instruction_late
    after_retrieval_instruction_late = len(tokenizer(llm_prompt).input_ids) - off_set

    llm_prompt += f'{query.strip()}'
    end_len = len(tokenizer(llm_prompt).input_ids) - off_set
    llm_prompt += prompt_suffix

    query_start_idx = start_len
    query_end_idx = end_len

    return llm_prompt, (doc_span, query_start_idx, after_retrieval_instruction_late, query_end_idx)


@pytest.mark.parametrize("model_id", TEST_MODELS)
def test_core_reranker(cache_dir, project_root, model_id):
    register_plugins()

    cfg = ensure_config_for_model(project_root, "core_reranker", model_id)

    llm = HookLLM(
        model=model_id,
        worker_name="probe_hook_qk",
        analyzer_name="core_reranker",
        config_file=str(cfg),
        download_dir=str(cache_dir),
        gpu_memory_utilization=0.5,
        dtype=torch.float16,
        enable_hook=True,
    )

    query = "Which city is older: Rome or New York?"
    documents = [
        ["Rome was founded in 753 BC."],
        ["New York was founded in 1624."],
    ]

    textQ, query_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model_id, query, documents)
    textNA, na_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model_id, "N/A", documents)

    llm.generate(textQ, temperature=0.1, max_tokens=1)
    llm.generate(textNA, cleanup=False, temperature=0.1, max_tokens=1)

    stats = llm.analyze(
        analyzer_spec={"query_spec": query_spec, "na_spec": na_spec}
    )

    assert "scores" in stats
    assert "ranking" in stats
    assert len(stats["scores"][0]) == len(documents)
    assert len(stats["ranking"][0]) == len(documents)
