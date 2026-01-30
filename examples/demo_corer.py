import os
import multiprocessing as mp
import torch
import time
from typing import List

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_hook_plugins import HookLLM

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

if __name__ == "__main__":

    cache_dir = "/dccstor/pyrite/irene/"
    model = 'mistralai/Mistral-7B-Instruct-v0.3' # 'ibm-granite/granite-3.1-8b-instruct'  # 'Qwen/Qwen2-1.5B-Instruct' #
    
    dtype_map = {
        'mistralai/Mistral-7B-Instruct-v0.3': torch.float16,
        'ibm-granite/granite-3.1-8b-instruct': torch.float16,
        'Qwen/Qwen2-1.5B-Instruct': torch.float
    }
    
    llm = HookLLM(
        model=model,
        worker_name="probe_hook_qk",
        analyzer_name="core_reranker",
        config_file=f'model_configs/core_reranker/{model.split("/")[-1]}.json',
        download_dir=cache_dir,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=dtype_map[model],
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_hook=True, 
        tensor_parallel_size=1  # the number of gpus
    )
        
    test_cases = [
        {
            "query": "Which magazine was started first Arthur's Magazine or First for Women?",
            "documents": [
                [
                "Radio City is India's first private FM radio station and was started on 3 July 2001.",
                " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
                " It plays Hindi, English and regional songs.",
                " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.",
                " Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.",
                " The Radio station currently plays a mix of Hindi and Regional music.",
                " Abraham Thomas is the CEO of the company."
                ],
                [
                "Football in Albania existed before the Albanian Football Federation (FSHF) was created.",
                " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .",
                " Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.",
                " In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA."
                ],
                [
                "Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
                " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
                " Echosmith started first as \"Ready Set Go!\"",
                " until they signed to Warner Bros.",
                " Records in May 2012.",
                " They are best known for their hit song \"Cool Kids\", which reached number 13 on the \"Billboard\" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.",
                " The song was Warner Bros.",
                " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.",
                " The band's debut album, \"Talking Dreams\", was released on October 8, 2013."
                ],
                [
                "Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.",
                " Many started first as girls' seminaries or academies.",
                " Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.",
                " Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level."
                ],
                [
                "The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum."
                ],
                [
                "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.",
                " Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.",
                " In May 1846 it was merged into \"Godey's Lady's Book\"."
                ],
                [
                "The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.",
                " Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.",
                " Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.",
                " The regular season included just 12 rounds, where all the teams went to the semifinals.",
                " In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk."
                ],
                [
                "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
                " The magazine was started in 1989.",
                " It is based in Englewood Cliffs, New Jersey.",
                " In 2011 the circulation of the magazine was 1,310,696 copies."
                ],
                [
                "The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.",
                " The fire started as two separate fires on November 15, 2008.",
                " The \"Freeway Fire\" started first shortly after 9am with the \"Landfill Fire\" igniting approximately 2 hours later.",
                " These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda."
                ],
                [
                "William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.",
                " It is most known for their premium jeans.",
                " On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.",
                " The label also produces other clothing items such as jackets and tops.",
                " The company started first as a denim line, later evolving into a men’s and women’s clothing line."
                ]
            ]
        },
        {
            "query": "Which came first, the invention of the telephone or the light bulb?",
            "documents": [
                [
                "Alexander Graham Bell is credited with inventing the first practical telephone.",
                " He was awarded the U.S. patent for the invention of the telephone on March 7, 1876.",
                " The first successful demonstration of the telephone took place shortly thereafter, when Bell famously called his assistant, saying, 'Mr. Watson, come here, I want to see you.'",
                " Bell’s invention revolutionized communication by allowing people to talk to each other over long distances."
                ],
                [
                "Thomas Edison is widely known for inventing the first commercially practical incandescent light bulb.",
                " Although he did not invent the concept of the light bulb itself, Edison developed a version that was safe, affordable, and long-lasting.",
                " His patent for the electric light bulb was filed in 1879, three years after Bell’s telephone patent.",
                " Edison's innovation led to widespread use of electric lighting and helped usher in the modern electrical age."
                ],
                [
                "Before Edison, several inventors worked on early versions of the light bulb.",
                " Sir Humphry Davy created the first electric arc lamp in the early 1800s, and later inventors like Joseph Swan in Britain improved upon the design.",
                " However, these early bulbs were inefficient or burned out quickly, and it was Edison who perfected the design for everyday use."
                ],
                [
                "The telephone was invented before the practical light bulb.",
                " Bell’s patent for the telephone was issued in 1876, while Edison’s patent for the light bulb was filed in 1879.",
                " Thus, the telephone came first."
                ],
                [
                "Both the telephone and the light bulb are considered groundbreaking inventions of the late 19th century.",
                " The telephone transformed communication, while the light bulb transformed how people lived and worked at night.",
                " Together, they symbolize the rapid technological progress of that era."
                ],
                [
                "Edison and Bell were contemporaries and pioneers of the Second Industrial Revolution.",
                " Their inventions marked major milestones in human history, driving the growth of telecommunications and electrical infrastructure."
                ],
                [
                "In summary, the telephone was invented in 1876 and the light bulb in 1879.",
                " Therefore, the invention of the telephone came first."
                ]
            ]
        }
    ]
        
    for case in test_cases:
        print("=" * 50)
        query = case["query"]
        documents = case["documents"]
        
        # Apply chat template and get ranges
        text, query_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model, query, documents)
        llm.generate(text, temperature=0.1, max_tokens=1)
        
        text, na_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model, 'N/A', documents)
        llm.generate(text, cleanup=False, temperature=0.1, max_tokens=1)
        
        stats = llm.analyze(analyzer_spec={'query_spec': query_spec, 'na_spec': na_spec})
        print(f"Sorted document IDs and scores by CoRe-Reranking: {stats['ranking']}: {stats['scores']}")

        llm.llm_engine.reset_prefix_cache()
        # # Runtime comparison with vllm without hooks
        # llm.generate(text, temperature=0.1, max_tokens=1, use_hook=False)
        # llm.llm_engine.reset_prefix_cache()


    ### batch processing, beta mode, not fully tested
    print("=" * 50)
    print("Batch processing examples...")
    text_querys = []
    query_specs = []
    text_nas = []
    na_specs = []
    for case in test_cases:
        query = case["query"]
        documents = case["documents"]
        
        # Apply chat template and get ranges
        text_query, query_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model, query, documents)
        text_na, na_spec = apply_chat_template_and_get_ranges(llm.tokenizer, model, 'N/A', documents)

        text_querys.append(text_query)
        query_specs.append(query_spec)        
        text_nas.append(text_na)
        na_specs.append(na_spec)
    
    llm.generate(text_querys, temperature=0.1, max_tokens=1)
    llm.generate(text_nas, cleanup=False, temperature=0.1, max_tokens=1)
    
    stats = llm.analyze(analyzer_spec={'query_spec': query_specs, 'na_spec': na_specs})
    print(f"Sorted document IDs and scores by CoRe-Reranking: {stats['ranking']}: {stats['scores']}")
    llm.llm_engine.reset_prefix_cache()