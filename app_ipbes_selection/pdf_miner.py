from datetime import datetime
import asyncio
import logging
import functools
from pathlib import Path
import traceback

from tqdm import tqdm
from pdfminer.high_level import extract_text
from llm_analyzer import safe_openai_completion


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
    return candidate_filepath


@log_errors()
async def clean_nomination_pdf(pdf_path, output_dir):
    print(pdf_path)
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
    formatted_pdf_text = await safe_openai_completion(messages, "gpt-5-nano")
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
    name = await safe_openai_completion(messages, "gpt-5-nano")
    target_path = output_dir / Path(f"recommendation_{name}.txt")
    write_with_suffix_increment(target_path, formatted_pdf_text)


@log_errors()
async def clean_cv_pdf(pdf_path, output_dir):
    print(pdf_path)
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
    formatted_cv_text = await safe_openai_completion(messages, "gpt-5-nano")
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
    name = await safe_openai_completion(messages, "gpt-5-nano")
    target_path = output_dir / Path(f"cv_{name}.txt")
    write_with_suffix_increment(target_path, formatted_cv_text)


async def main():
    input_dir = Path("pdf_data")
    target_dir = Path("txts")
    candidate_target_dir = Path(target_dir)
    count = 1

    while candidate_target_dir.exists():
        candidate_target_dir = target_dir.with_name(
            f"{target_dir.name}_{count}"
        )
        count += 1
    target_dir = candidate_target_dir
    target_dir.mkdir(parents=True, exist_ok=False)

    task_list = []

    for nomination_pdf_path in input_dir.glob("ipbes_noms/*.pdf"):
        task_list.append(
            asyncio.create_task(
                clean_nomination_pdf(nomination_pdf_path, target_dir)
            )
        )
        break

    for cvs_pdf_path in input_dir.glob("ipbes_cvs/*.pdf"):
        task_list.append(
            asyncio.create_task(clean_cv_pdf(cvs_pdf_path, target_dir))
        )
        break

    for task in tqdm(
        asyncio.as_completed(task_list),
        total=len(task_list),
        desc="cleaning pdfs",
    ):
        await task


if __name__ == "__main__":
    asyncio.run(main())
