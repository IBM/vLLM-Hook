"""Single-example AttnLink-U schema linking with the stock QK worker."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
CONFIG = ROOT / "model_configs/attnlink/Qwen2.5-Coder-7B-Instruct.json"
# BIRD development example 128; the example data in this block is CC BY-SA 4.0.
# Source: https://bird-bench.github.io/ (Li et al., 2023), via Songjw133/AttnLink.
# Code remains under the repository license. See docs/use_cases/attnlink.md.
INPUT_SEQ = """Task Overview:
You are a data science expert. Your task is column-level schema linking. Given a natural language question and a database schema, identify the gold database tables and columns needed to write the SQL query for the question.

Database Engine:
SQLite

Database Schema:
CREATE TABLE account (
    account_id integer, -- Example: [1]
    district_id integer, -- location of branch, Example: [18]
    frequency text, -- Example: ['POPLATEK MESICNE']
    date date, -- Example: ['1995-03-24']
    PRIMARY KEY (account_id),
    CONSTRAINT fk_account_district_id FOREIGN KEY (district_id) REFERENCES district (district_id)
);

CREATE TABLE card (
    card_id integer, -- credit card id, Example: [1]
    disp_id integer, -- disposition id, Example: [9]
    type text, -- Example: ['gold']
    issued date, -- Example: ['1998-10-16']
    PRIMARY KEY (card_id),
    CONSTRAINT fk_card_disp_id FOREIGN KEY (disp_id) REFERENCES disp (disp_id)
);

CREATE TABLE client (
    client_id integer, -- Example: [1]
    gender text, -- Example: ['F']
    birth_date date, -- Example: ['1970-12-13']
    district_id integer, -- location of branch, Example: [18]
    PRIMARY KEY (client_id),
    CONSTRAINT fk_client_district_id FOREIGN KEY (district_id) REFERENCES district (district_id)
);

CREATE TABLE disp (
    disp_id integer, -- disposition id, Example: [1]
    client_id integer, -- Example: [1]
    account_id integer, -- Example: [1]
    type text, -- Example: ['OWNER']
    PRIMARY KEY (disp_id),
    CONSTRAINT fk_disp_client_id FOREIGN KEY (client_id) REFERENCES client (client_id),
    CONSTRAINT fk_disp_account_id FOREIGN KEY (account_id) REFERENCES account (account_id)
);

CREATE TABLE district (
    district_id integer, -- location of branch, Example: [1]
    A2 text, -- district_name, Example: ['Hl.m. Praha']
    A3 text, -- region, Example: ['Prague']
    A4 text, -- number of inhabitants, Example: ['1204953']
    A5 text, -- no. of municipalities with inhabitants < 499, Example: ['0']
    A6 text, -- no. of municipalities with inhabitants 500-1999, Example: ['0']
    A7 text, -- no. of municipalities with inhabitants 2000-9999, Example: ['0']
    A8 integer, -- no. of municipalities with inhabitants > 10000, Example: [1]
    A9 integer, -- Example: [1]
    A10 real, -- ratio of urban inhabitants, Example: [100.0]
    A11 integer, -- average salary, Example: [12541]
    A12 real, -- unemployment rate 1995, Example: [0.2]
    A13 real, -- unemployment rate 1996, Example: [0.43]
    A14 integer, -- no. of entrepreneurs per 1000 inhabitants, Example: [167]
    A15 integer, -- no. of committed crimes 1995, Example: [85677]
    A16 integer, -- no. of committed crimes 1996, Example: [99107]
    PRIMARY KEY (district_id)
);

CREATE TABLE loan (
    loan_id integer, -- Example: [4959]
    account_id integer, -- Example: [2]
    date date, -- Example: ['1994-01-05']
    amount integer, -- Example: [80952]
    duration integer, -- Example: [24]
    payments real, -- monthly payments, Example: [3373.0]
    status text, -- Example: ['A']
    PRIMARY KEY (loan_id),
    CONSTRAINT fk_loan_account_id FOREIGN KEY (account_id) REFERENCES account (account_id)
);

CREATE TABLE `order` (
    order_id integer, -- Example: [29401]
    account_id integer, -- Example: [1]
    bank_to text, -- bank of the recipient, Example: ['YZ']
    account_to integer, -- account of the recipient, Example: [87144583]
    amount real, -- debited amount, Example: [2452.0]
    k_symbol text, -- characterization of the payment, Example: ['SIPO']
    PRIMARY KEY (order_id),
    CONSTRAINT fk_order_account_id FOREIGN KEY (account_id) REFERENCES account (account_id)
);

CREATE TABLE trans (
    trans_id integer, -- transaction id, Example: [1]
    account_id integer, -- Example: [1]
    date date, -- date of transaction, Example: ['1995-03-24']
    type text, -- +/- transaction, Example: ['PRIJEM']
    operation text, -- mode of transaction, Example: ['VKLAD']
    amount integer, -- amount of money, Example: [1000]
    balance integer, -- balance after transaction, Example: [1000]
    k_symbol text, -- characterization of the transaction, Example: ['SIPO']
    bank text, -- bank of the partner, Example: ['AB']
    account integer, -- account of the partner, Example: [41403269]
    PRIMARY KEY (trans_id),
    CONSTRAINT fk_trans_account_id FOREIGN KEY (account_id) REFERENCES account (account_id)
);
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
A2 refers to districts; Female refers to gender = 'F'
List the top nine districts, by descending order, from the highest to the lowest, the number of female account holders.

Instructions:

* Your task is column-level schema linking.
* First, identify all and only the relevant columns required by the gold SQL query.
* Answer only using candidate column identifiers from Candidate Columns.
* Each candidate column identifier is written in exact `column@table` format.
* You must copy the relevant candidate identifier exactly as it appears in Candidate Columns.
* Your final answer must contain exactly one `column@table` identifier.
* If multiple candidate columns are relevant, choose one relevant identifier uniformly at random.
* Do not prefer the first, last, most obvious, or most important relevant column.
* Do not output SQL, explanations, reasoning, table names alone, column names alone, punctuation, bullets, numbering, spaces, or any other text.
* Output only the single selected `column@table` identifier.

Example:
Question:
What is the maximum price of products in the Electronics category?

Candidate Columns:
#table: products (
product_id@products
category@products
price@products
product_name@products
)

Relevant candidate identifiers:
category@products
price@products

Valid final answer:
category@products

Also valid final answer:
price@products

**Now start your choice**:

Candidate Columns:
#table: account (
account_id@account
district_id@account
frequency@account
date@account
)
#table: card (
card_id@card
disp_id@card
type@card
issued@card
)
#table: client (
client_id@client
gender@client
birth_date@client
district_id@client
)
#table: disp (
disp_id@disp
client_id@disp
account_id@disp
type@disp
)
#table: district (
district_id@district
a2@district
a3@district
a4@district
a5@district
a6@district
a7@district
a8@district
a9@district
a10@district
a11@district
a12@district
a13@district
a14@district
a15@district
a16@district
)
#table: loan (
loan_id@loan
account_id@loan
date@loan
amount@loan
duration@loan
payments@loan
status@loan
)
#table: `order` (
order_id@order
account_id@order
bank_to@order
account_to@order
amount@order
k_symbol@order
)
#table: trans (
trans_id@trans
account_id@trans
date@trans
type@trans
operation@trans
amount@trans
balance@trans
k_symbol@trans
bank@trans
account@trans
)

"""

POSITIVE_COLS = ['client.client_id',
 'client.gender',
 'client.district_id',
 'district.district_id',
 'district.A2']
SOURCE = {'conversion': 'Unchanged input_seq, positive_cols, positive_tables and SQL from the '
               'AttnLink artifact; includes schema descriptions, copying instructions '
               'and column@table candidates.',
 'dataset': 'BIRD development split',
 'dataset_sha256': 'b3a3631495508a90eead7c0caaf92a63466a37ebeefa99ab8aa4b4a5a5009b1e',
 'derived_from': 'https://github.com/Songjw133/AttnLink',
 'index': 128,
 'license': 'CC-BY-SA-4.0',
 'url': 'https://bird-bench.github.io/'}



def prepare_prompt(tokenizer, input_seq):
    """Map identifiers in the final candidate block to full-prompt token spans."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_seq}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    marker = "Candidate Columns:\n"
    block = input_seq.rfind(marker)
    if block < 0:
        raise ValueError("Missing final Candidate Columns block.")
    cursor = block + len(marker)
    content_start = rendered.index(input_seq)
    candidates, spans = [], []
    for line in input_seq[cursor:].splitlines(keepends=True):
        item = line.strip()
        if item and item != ")" and not item.startswith("#table:"):
            if "@" not in item:
                raise ValueError(f"Invalid candidate line: {item!r}")
            start = content_start + cursor + line.index(item)
            end = start + len(item)
            covered = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
            if not covered or offsets[covered[0]][0] > start or offsets[covered[-1]][1] < end:
                raise ValueError(f"Cannot align candidate {item!r} to tokens.")
            candidates.append(item)
            spans.append((covered[0], covered[-1] + 1))
        cursor += len(line)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Final candidate block must contain unique identifiers.")
    return ids, {"candidates": candidates, "candidate_spans": spans,
                 "prompt_length": len(ids)}


def column_ref(candidate):
    column, separator, table = candidate.rpartition("@")
    if not separator or not column or not table:
        raise ValueError(f"Invalid column@table identifier: {candidate!r}")
    return f"{table}.{column}".casefold()


def gold_ref(value):
    table, column = value.split(".", 1)
    return ".".join(part.strip().strip('`"[]').casefold() for part in (table, column))


def evaluate_ranking(candidates, ranking, positive_cols):
    """Average precision over the complete ranking, with input-order tie breaks."""
    refs = [column_ref(c) for c in candidates]
    gold = {gold_ref(c) for c in positive_cols}
    if len(set(refs)) != len(refs):
        raise ValueError("Candidate identifiers map to duplicate column references.")
    if not gold or not gold.issubset(refs):
        raise ValueError("Gold columns must be nonempty and present in the candidates.")
    if sorted(ranking) != list(range(len(candidates))):
        raise ValueError("Ranking must contain every candidate exactly once.")
    hits, total = 0, 0.0
    for rank, i in enumerate(ranking, 1):
        if refs[i] in gold:
            hits += 1
            total += hits / rank
    return total / len(gold), [ref in gold for ref in refs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help="Model ID or local path to the same model.")
    parser.add_argument("--out-dir", type=Path, default=None, help="New directory for this run.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Candidate-distribution temperature (not generation temperature).")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="Select the shortest prefix reaching this candidate probability mass.")
    args = parser.parse_args()
    from vllm_hook_plugins.analyzers.attnlink_analyzer import select_columns
    # Validate selection settings before loading the model.
    select_columns([1.0], args.temperature, args.top_p)
    out_dir = args.out_dir or Path("cache") / ("attnlink_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    out_dir.mkdir(parents=True, exist_ok=False)

    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    llm = HookLLM(
        model=args.model, worker_name="probe_hook_qk", analyzer_name="attnlink",
        config_file=str(CONFIG), hook_dir=str(out_dir / "hooks"),
        dtype="bfloat16", tensor_parallel_size=1, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        max_model_len=4096, max_num_batched_tokens=4096, max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        ids, spec = prepare_prompt(llm.tokenizer, INPUT_SEQ)
        if len(ids) + 1 > 4096:
            raise ValueError("Sample exceeds the demo's 4096-token context limit.")
        outputs = llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0.0, max_tokens=1, seed=0),
            save_to_disk=False,
        )
        if outputs[0].prompt_token_ids != ids:
            raise RuntimeError("Inference token IDs differ from span-alignment token IDs.")
        probes = getattr(outputs[0], "probes", None)
        if probes is None:
            raise RuntimeError("QK probes are missing; check the stock Hook installation.")
        spec.update(temperature=args.temperature, top_p=args.top_p)
        result = llm.analyze(analyzer_spec=spec, probes=probes)
        ap, gold = evaluate_ranking(spec["candidates"], result["ranking"], POSITIVE_COLS)
        selected = set(result["selected"])
        ranking, cumulative = [], 0.0
        for rank, i in enumerate(result["ranking"], 1):
            cumulative += result["probabilities"][i]
            ranking.append({"rank": rank, "column": column_ref(spec["candidates"][i]),
                            "candidate": spec["candidates"][i], "score": result["scores"][i],
                            "probability": result["probabilities"][i],
                            "cumulative_probability": cumulative,
                            "gold": gold[i], "selected": i in selected})
        hits = sum(gold[i] for i in selected)
        precision, recall = hits / len(selected), hits / sum(gold)
        selection = {"temperature": args.temperature, "top_p": args.top_p,
                     "selected_columns": [row["column"] for row in ranking if row["selected"]],
                     "selected_count": len(selected), "gold_count": sum(gold), "true_positives": hits,
                     "probability_mass": ranking[len(selected) - 1]["cumulative_probability"],
                     "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if hits else 0.0}
        question = INPUT_SEQ.split("\nQuestion:\n", 1)[1].split("\n\nInstructions:", 1)[0].strip()
        report = {"question": question, "source": SOURCE, "model": args.model,
                  "layer": 22, "head": 12, "pooling": "span_mean", "dtype": "bfloat16",
                  "execution": "eager", "prompt_tokens": len(ids), "average_precision": ap,
                  "input_sha256": hashlib.sha256(INPUT_SEQ.encode()).hexdigest(),
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("vllm", "vllm-hook-plugins", "torch", "transformers")},
                  "selection": selection, "ranking": ranking}
        (out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nQuestion: {question}\nModel: {MODEL} | layer 22 / head 12")
        print(f"Prompt tokens: {len(ids)} | Candidates: {len(ranking)} | Gold: {sum(gold)}")
        print(f"Temperature: {args.temperature:g} | Top-p: {args.top_p:g}")
        print(f"{'Rank':>4}  {'Column':<28}  {'Prob.':>8}  {'Cum.':>8}  Gold  Selected")
        for row in ranking:
            print(f"{row['rank']:>4}  {row['column']:<28}  {row['probability']:>8.4%}  "
                  f"{row['cumulative_probability']:>8.4%}    {'*' if row['gold'] else '-'}      "
                  f"{'*' if row['selected'] else '-'}")
        print("\nSelected columns: " + ", ".join(selection["selected_columns"]))
        print(f"Selected: {len(selected)}/{len(ranking)} | Gold covered: {hits}/{sum(gold)} | "
              f"Selected mass: {selection['probability_mass']:.4%}")
        print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | "
              f"F1: {selection['f1']:.2%} | AP: {ap:.6f}")
        print(f"Full result: {out_dir / 'result.json'}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
