"""Mine recommendation and CV pdfs for LLM processing."""

from typing import Dict, List, Callable
import json
from collections import defaultdict
import sys
from datetime import datetime
import asyncio
import logging
import functools
from pathlib import Path
import traceback
import re
from collections import namedtuple

from tqdm import tqdm
from pdfminer.high_level import extract_text
from llm_analyzer import safe_openai_completion, estimate_tokens_for_messages

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        "%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s"
        " [%(funcName)s:%(lineno)d] %(message)s"
    ),
)
LOGGER = logging.getLogger(__name__)

logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

ERROR_LOG = Path("file_errors.txt")
_ERROR_LOG_LOCK = asyncio.Lock()


def log_errors(error_log: Path = ERROR_LOG):
    def _decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _awrap(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # attempt to find a Path-like input to log (e.g., the pdf path)
                    target_path = next(
                        (a for a in args if isinstance(a, Path)), None
                    )
                    async with _ERROR_LOG_LOCK:
                        with open(error_log, "a", encoding="utf-8") as f:
                            f.write(
                                f"[{datetime.utcnow().isoformat()}Z] {func.__name__} "
                                f"target={target_path!s} error={repr(e)}\n"
                            )
                            f.write(traceback.format_exc())
                            f.write("\n---\n")
                    return None

            return _awrap
        else:

            @functools.wraps(func)
            def _wrap(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    target_path = next(
                        (a for a in args if isinstance(a, Path)), None
                    )
                    # synchronous guard for completeness
                    with open(error_log, "a", encoding="utf-8") as f:
                        f.write(
                            f"[{datetime.utcnow().isoformat()}Z] {func.__name__} "
                            f"target={target_path!s} error={repr(e)}\n"
                        )
                        f.write(traceback.format_exc())
                        f.write("\n---\n")
                    return None

            return _wrap

    return _decorator


def write_with_suffix_increment(target_path: Path, content: str):
    candidate_filepath = Path(target_path)
    count = 1
    while candidate_filepath.exists():
        candidate_filepath = target_path.with_name(
            f"{target_path.stem}_{count}{target_path.suffix}"
        )
        count += 1

    candidate_filepath.write_text(content, encoding="utf-8")
    LOGGER.info(f"writing {str(target_path)}")
    return candidate_filepath


@log_errors()
async def clean_nomination_pdf(pdf_path, output_dir):
    pdf_text = extract_text(str(pdf_path))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly skilled document formatter. "
                "Your task is to take raw text extracted from a PDF, which may contain awkward line breaks, "
                "page numbers, table fragments, and formatting artifacts, and reconstruct it into a clean, "
                "readable, and well-structured plain text document. "
                "Preserve all important headings, sections, lists, tables, and metadata. "
                "Merge broken lines into proper paragraphs, fix spacing, and remove duplicate or irrelevant page markers. "
                "Do not omit meaningful content, and keep the original meaning intact. "
                "Output should be neatly formatted, easy to read, and suitable for professional use."
            ),
        },
        {
            "role": "user",
            "content": f"{pdf_text}",
        },
    ]
    formatted_pdf_text = await safe_openai_completion(
        messages, "gpt-5-nano", job_id=f"nomination {pdf_path}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant. "
                "Your only task is to identify the full name of the person receiving the recommendation "
                "from the given text. Respond with only the name, with no extra words or punctuation. "
                "If you are unsure, respond with 'UNKNOWN'."
            ),
        },
        {
            "role": "user",
            "content": f"{formatted_pdf_text}",
        },
    ]
    name = await safe_openai_completion(
        messages, "gpt-5-nano", job_id=f"nomination NAME {pdf_path}"
    )
    target_path = output_dir / Path(f"recommendation_{name}.txt")
    write_with_suffix_increment(target_path, formatted_pdf_text)


@log_errors()
async def clean_cv_pdf(pdf_path, output_dir):
    cv_text = extract_text(str(pdf_path))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional document formatter. "
                "The user will provide the raw text of a CV that may contain broken lines, bad spacing, "
                "page numbers, or other PDF extraction artifacts. "
                "Your task is to reconstruct the CV into a clean, logically structured plain-text document. "
                "Preserve the original information and section headings (e.g., Education, Experience, Skills, Publications), "
                "merging broken lines into full sentences where appropriate. "
                "Remove any duplicate headers, stray page numbers, or irrelevant artifacts. "
                "Do not add extra commentary, and do not omit meaningful content. "
                "Output only the cleaned, formatted CV in plain text."
            ),
        },
        {
            "role": "user",
            "content": f"{cv_text}",
        },
    ]
    formatted_cv_text = await safe_openai_completion(
        messages, "gpt-5-nano", job_id=f"cv {pdf_path}"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant. "
                "The user will provide the text of a CV. "
                "Your task is to identify the full name of the person the CV belongs to. "
                "Respond with only the full name, with no extra words, punctuation, or formatting. "
                "If you are unsure, respond with 'UNKNOWN'."
            ),
        },
        {
            "role": "user",
            "content": f"{cv_text}",
        },
    ]
    name = await safe_openai_completion(
        messages, "gpt-5-nano", job_id=f"cv NAME {pdf_path}"
    )
    target_path = output_dir / Path(f"cv_{name}.txt")
    write_with_suffix_increment(target_path, formatted_cv_text)


def split_files_by_prefix(directory: Path):
    recommendations = defaultdict(list)
    cvs = defaultdict(list)

    rec_pattern = re.compile(r"^recommendation_(.+?)(?:_\d+)?\.txt$")
    cv_pattern = re.compile(r"^cv_(.+?)(?:_\d+)?\.txt$")

    for file_path in Path(directory).iterdir():
        if not file_path.is_file():
            continue

        m_rec = rec_pattern.match(file_path.name)
        if m_rec:
            name = m_rec.group(1)
            recommendations[name].append(file_path)
            continue

        m_cv = cv_pattern.match(file_path.name)
        if m_cv:
            name = m_cv.group(1)
            cvs[name].append(file_path)
            continue

    return dict(recommendations), dict(cvs)


async def distangle_names(file_dir: Path, json_target_path: Path):
    recommendation_names, cv_names = split_files_by_prefix(file_dir)

    distangle_name_messages = [
        {
            "role": "system",
            "content": (
                "You are an entity resolution assistant. Your job is to CLUSTER person names from two lists "
                "into sets of variants that refer to the same person.\n"
                "\n"
                "Normalization & matching rules:\n"
                "- Ignore honorifics/titles (Dr., Prof., Mr., Ms.), degrees (PhD), and suffixes (Jr., III).\n"
                "- Normalize case, whitespace, punctuation, and diacritics (Ordóñez == Ordonez).\n"
                "- Consider nicknames and diminutives (Ale == Alejandro), initials (A. == Alejandro), "
                "middle names/initials (optional/missing), and order variations.\n"
                "- Prefer matches with the same surname; be cautious with common surnames.\n"
                "- If uncertain (weak or conflicting evidence), do NOT force a match.\n"
                "\n"
                "Alpha (canonical) name selection:\n"
                "- Choose a single canonical name per cluster ('alpha') by these tie-breakers in order:\n"
                "  1) Most complete form (given + middle + surname).\n"
                "  2) Appears as a full form in either list.\n"
                "  3) If still tied, pick the version from List B; if still tied, pick lexicographically smallest.\n"
                "\n"
                "Output STRICT JSON only (no prose). Use this schema:\n"
                "{\n"
                '  "clusters": [\n'
                "    {\n"
                '      "alpha": "Full Canonical Name",\n'
                '      "aliases": ["variant1", "variant2", ...],\n'
                '      "members": {\n'
                '        "A": [ { "index": int, "name": "..." }, ... ],\n'
                '        "B": [ { "index": int, "name": "..." }, ... ]\n'
                "      },\n"
                '      "confidence": 0.0\n'
                "    }\n"
                "  ],\n"
                '  "unmatched": {\n'
                '    "A": [ { "index": int, "name": "..." }, ... ],\n'
                '    "B": [ { "index": int, "name": "..." }, ... ]\n'
                "  }\n"
                "}\n"
                "\n"
                "Notes:\n"
                "- Include every item from both lists exactly once, either in a cluster or in unmatched.\n"
                "- aliases = unique set of all spellings seen for the cluster (from both lists), excluding the alpha form.\n"
                "- confidence in [0,1] reflects how certain the cluster represents the same person.\n"
                "- Do not invent names. Do not add commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                "List A (index then name, one per line):\n"
                + "\n".join(
                    f"{i}. {name}"
                    for i, name in enumerate(recommendation_names)
                )
                + "\n\n"
                "List B (index then name, one per line):\n"
                + "\n".join(f"{i}. {name}" for i, name in enumerate(cv_names))
                + "\n\n"
                "Cluster the names across both lists according to the rules and return STRICT JSON."
            ),
        },
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_resolution_result",
            "schema": {
                "type": "object",
                "properties": {
                    "clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "alpha": {"type": "string"},
                                "aliases": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "members": {
                                    "type": "object",
                                    "properties": {
                                        "A": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {
                                                        "type": "integer"
                                                    },
                                                    "name": {"type": "string"},
                                                },
                                                "required": ["index", "name"],
                                                "additionalProperties": False,
                                            },
                                        },
                                        "B": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "index": {
                                                        "type": "integer"
                                                    },
                                                    "name": {"type": "string"},
                                                },
                                                "required": ["index", "name"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["A", "B"],
                                    "additionalProperties": False,
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                            "required": [
                                "alpha",
                                "aliases",
                                "members",
                                "confidence",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "unmatched": {
                        "type": "object",
                        "properties": {
                            "A": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                    "required": ["index", "name"],
                                    "additionalProperties": False,
                                },
                            },
                            "B": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                    "required": ["index", "name"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["A", "B"],
                        "additionalProperties": False,
                    },
                },
                "required": ["clusters", "unmatched"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
    formatted_names_content = await safe_openai_completion(
        distangle_name_messages,
        "o3",
        job_id=f"distangle names",
        response_format=response_format,
    )
    print(formatted_names_content)
    try:
        data = json.loads(formatted_names_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}")

    # write to file
    output_path = Path(json_target_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _format_block(name: str, contexts: List[str]) -> str:
    if not isinstance(contexts, list):
        contexts = [str(contexts)]
    joined = "\n\n".join(str(c) for c in contexts if c)
    return f"---\nName: {name}\nContext:\n{joined}\n---\n"


def _base_messages(
    system_content: str, user_question: str, preface: str = ""
) -> List[Dict[str, str]]:
    preface = preface or "Selection question:\n{q}\n\nDataset context blocks:\n"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": preface.format(q=user_question)},
    ]


def _messages_with_blocks(
    system_content: str, user_question: str, blocks_text: str, preface: str = ""
) -> List[Dict[str, str]]:
    msgs = _base_messages(system_content, user_question, preface=preface)
    msgs[-1]["content"] = msgs[-1]["content"] + blocks_text
    return msgs


def chunk_people_into_batches(
    system_content: str,
    user_question: str,
    name_to_context: Dict[str, List[str]],
    token_limit: int,
    estimate_tokens_for_messages: Callable[[List[Dict[str, str]]], int],
    reserved_for_response: int = 1024,
    preface: str = "Selection question:\n{q}\n\nDataset context blocks:\n",
) -> List[List[Dict[str, str]]]:
    base_msgs = _base_messages(system_content, user_question, preface=preface)
    max_allowed = max(0, token_limit - reserved_for_response)

    batches: List[List[Dict[str, str]]] = []
    current_blocks: List[str] = []
    current_msgs: List[Dict[str, str]] = base_msgs
    current_tokens = estimate_tokens_for_messages(current_msgs)

    for name, ctxs in tqdm(name_to_context.items()):
        block = _format_block(name, ctxs)
        candidate_msgs = _messages_with_blocks(
            system_content,
            user_question,
            "".join(current_blocks) + block,
            preface=preface,
        )
        candidate_tokens = estimate_tokens_for_messages(candidate_msgs)

        if candidate_tokens <= max_allowed:
            current_blocks.append(block)
            current_msgs = candidate_msgs
            current_tokens = candidate_tokens
        else:
            if current_blocks:
                batches.append(current_msgs)
            LOGGER.info(f"new batch with {name}")
            # start a new batch with this block alone (if it fits)
            new_msgs = _messages_with_blocks(
                system_content, user_question, block, preface=preface
            )
            if estimate_tokens_for_messages(new_msgs) <= max_allowed:
                current_blocks = [block]
                current_msgs = new_msgs
                current_tokens = estimate_tokens_for_messages(current_msgs)
            else:
                # skip oversized block that cannot fit even by itself
                current_blocks = []
                current_msgs = base_msgs
                current_tokens = estimate_tokens_for_messages(current_msgs)

    if current_blocks:
        batches.append(current_msgs)
    LOGGER.info(f"returning {len(batches)} batches")
    return batches


SELECTION_COMMITTEE_BASE_MESSAGE = (
    "You help a selection committee identify candidates from provided CV/recommendation excerpts. "
    "Each request contains ONLY a subset (batch) of people; do not assume anything about people not shown. "
    "It is acceptable for this batch to have zero relevant candidates, or for all of them to be relevant.\n"
    "\n"
    "Ground rules:\n"
    "- Use ONLY the provided context blocks; do not invent facts.\n"
    "- Treat each name as a distinct person unless evidence shows they are aliases.\n"
    "- Prefer direct, specific evidence over vague claims; include verbatim snippets as evidence.\n"
    "- If evidence is weak or ambiguous, reflect that with lower scores and explicit notes.\n"
    "- Your output must be valid JSON and follow the schema below. The shortlist may be empty.\n"
    "- Place every person from this batch into exactly one of: shortlist, near_misses, or unknown_or_insufficient.\n"
    "\n"
    "Scoring rubric (0–100):\n"
    "  Relevance (0–40), Specificity (0–25), Recency/continuity (0–20), Contextual breadth (0–15).\n"
    "\n"
    "Output STRICT JSON only, no prose, with this schema:\n"
    "{\n"
    '  "query": str,\n'
    '  "batch_hint": str,  // optional identifier you are given (if any)\n'
    '  "shortlist": [\n'
    "    {\n"
    '      "name": str,\n'
    '      "score": int,\n'
    '      "summary": str,\n'
    '      "evidence": [ { "source": str, "snippet": str } ],\n'
    '      "fit_tags": [str],\n'
    '      "confidence": float\n'
    "    }\n"
    "  ],\n"
    '  "near_misses": [\n'
    '    { "name": str, "reason": str, "evidence": [ { "source": str, "snippet": str } ] }\n'
    "  ],\n"
    '  "unknown_or_insufficient": [str],\n'
    '  "notes": str\n'
    "}\n"
    "\n"
    "Citations:\n"
    '- Each evidence item must include "source" (e.g., "cv_Name.txt", "recommendation_Name.txt").\n'
    '- "snippet" should be a short verbatim excerpt (<300 chars).\n'
    "\n"
    'If none in this batch are relevant, return an empty "shortlist" and explain briefly in "notes".'
)


async def main():
    input_dir = Path("pdf_data")
    text_dir = Path("txts")

    # while candidate_text_dir.exists():
    #     candidate_text_dir = text_dir.with_name(
    #         f"{text_dir.name}_{count}"
    #     )
    #     count += 1
    # text_dir = candidate_text_dir
    # text_dir.mkdir(parents=True, exist_ok=True)

    # task_list = []

    # LOGGER.info("creating clean noms")
    # for index, nomination_pdf_path in enumerate(
    #     input_dir.glob("ipbes_noms/*.pdf")
    # ):
    #     task_list.append(
    #         asyncio.create_task(
    #             clean_nomination_pdf(nomination_pdf_path, text_dir)
    #         )
    #     )

    # LOGGER.info("creating clean CVs")
    # for cvs_pdf_path in input_dir.glob("ipbes_cvs/*.pdf"):
    #     task_list.append(
    #         asyncio.create_task(clean_cv_pdf(cvs_pdf_path, text_dir))
    #     )

    # LOGGER.info("waiting for tasks to complete")
    # for task in tqdm(
    #     asyncio.as_completed(task_list),
    #     total=len(task_list),
    #     desc="cleaning pdfs",
    # ):
    #     await task

    #    await distangle_names(text_dir, "aliases.json")

    aliases = json.loads(open("aliases.json").read())
    alpha_to_alias = {}
    alias_to_alpha = {}

    for cluster in aliases["clusters"]:
        alpha = cluster["alpha"]
        aliases_set = set(cluster["aliases"]) | {
            alpha
        }  # include alpha as its own alias

        alpha_to_alias[alpha] = aliases_set
        for alias in aliases_set:
            alias_to_alpha[alias] = alpha
    name_to_context = defaultdict(list)

    for file_path in Path(text_dir).iterdir():
        if not file_path.is_file():
            continue
        _, name = file_path.stem.split("_")[0:2]
        if name in alias_to_alpha:
            name = alias_to_alpha[name]
        name_to_context[name].append(open(file_path, encoding="utf-8").read())
    open("name_to_context.json", "w", encoding="utf-8").write(
        json.dumps(name_to_context)
    )

    # user_question = "find me people who assess the ways in which Indigenous Peoples and local communities interact with their environment and the reciprocal relationships between nature and people, as well as the roles of social relationships, kinship, caring and the guardianship of nature and how these are supported by the knowledge systems of Indigenous Peoples and local communities, including values, practices, management, technologies and institutions for environmental and territorial governance"

    # batches = chunk_people_into_batches(
    #     system_content=SELECTION_COMMITTEE_BASE_MESSAGE,
    #     user_question=user_question,
    #     name_to_context=name_to_context,
    #     token_limit=200000,
    #     estimate_tokens_for_messages=estimate_tokens_for_messages,
    #     reserved_for_response=1500,
    # )

    # LOGGER.info("submitting for consideration")
    # result = await safe_openai_completion(batches[0], "gpt-5-nano")
    # open("result.json", "w", encoding="utf-8").write(json.dumps(result))
    # # open("batches.json", "w", encoding="utf-8").write(json.dumps(batches))


if __name__ == "__main__":
    asyncio.run(main())
