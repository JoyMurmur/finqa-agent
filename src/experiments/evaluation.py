"""
This sciprt evaluates selected agent architecture from the agent_candidates directory by
running experiments, computing accuracy metrics, and logging results and artifacts to MLflow.
"""

import asyncio
import os
import tempfile
import textwrap
from dataclasses import asdict
from pathlib import Path
from time import time

import dotenv
import matplotlib.pyplot as plt
import mlflow
import nest_asyncio
import numpy as np
import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.data import ConvFinQARecord, get_record, get_records
from src.experiments.evaluation_config import load_selected_agent_runtime
from src.logger import get_logger

nest_asyncio.apply()

dotenv.load_dotenv()
logger = get_logger(__name__)

# -------------- Configurations --------------
_config_dir = Path(__file__).parents[2] / "config"
prompts = yaml.safe_load((_config_dir / "prompts.yaml").read_text())


# -------------- Load Selected Agent Runtime --------------
agent_runtime = load_selected_agent_runtime(selected_agent="baseline")

initialize_chat = agent_runtime["initialize_chat"]
chat_turn = agent_runtime["chat_turn"]
async_chat_turn = agent_runtime["async_chat_turn"]
solver_config = agent_runtime["solver_config"]
reflector_config = agent_runtime["reflector_config"]

logger.info(
    "Selected agent variant: %s (%s)",
    agent_runtime["key"],
    agent_runtime["module_path"],
)


def run_single_record(record: ConvFinQARecord) -> tuple[dict, list[str], list[str]]:
    """Run the agent on a single record and return the final state, list of answers, and reasoning log."""
    state = initialize_chat(record)
    answers = []
    reasoning_log = []

    for i, question in enumerate(record.dialogue.conv_questions):
        logger.info(f"===== Question {i + 1}: {question} =====")
        state, answer = chat_turn(state, question)
        answers.append(answer)
        reasoning_log.append(state["solver"]["reasoning"])

        logger.info(f"Answer: {answer}")
        logger.info(f"Reasoning: {state['solver']['reasoning']}")

    return state, answers, reasoning_log


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def run_single_record_async(
    record: ConvFinQARecord,
) -> tuple[dict, list[str], list[str]]:
    """Run the agent on a single record asynchronously and return the final state, list of answers, and reasoning log."""
    state = initialize_chat(record)
    answers = []
    reasoning_log = []
    for question in record.dialogue.conv_questions:
        state, answer = await async_chat_turn(state, question)
        answers.append(answer)
        reasoning_log.append(state["solver"]["reasoning"])
    return state, answers, reasoning_log


async def run_records_async(
    records: list[ConvFinQARecord],
    max_concurrency: int = 10,
) -> list[tuple[dict, list[str], list[str], float]]:
    """Run the agent on multiple records asynchronously with a concurrency limit and return the results."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(record):
        async with semaphore:
            started = time()
            logger.info("start: %s", record.id)
            result = await run_single_record_async(record)
            elapsed = time() - started
            logger.info("end: %s (%.2f s)", record.id, elapsed)
        return (*result, elapsed)

    tasks = [run_with_semaphore(record) for record in records]
    return await asyncio.gather(*tasks)


def evaluate_question(
    predicted: str, ground_truth: str, tolerance: float = 1e-5
) -> bool:
    """Evaluate if the predicted answer matches the ground truth answer within a specified tolerance."""

    def to_float(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    p = to_float(predicted)
    g = to_float(ground_truth)

    return abs(p - g) <= tolerance


def evaluate_single_record(record: ConvFinQARecord, predicted: list[str]) -> dict:
    """Evaluate the predicted answers for a single record against the ground truths and return per-turn flags and overall accuracy."""
    ground_truths = record.dialogue.executed_answers

    turn_flags = [
        evaluate_question(p, g) for p, g in zip(predicted, ground_truths, strict=True)
    ]

    return {
        "record_id": record.id,
        "turn_flags": turn_flags,  # list of booleans indicating correctness of each turn
        "record_accuracy": sum(turn_flags) / len(turn_flags),  # per-record accuracy
    }


def inspect_response(
    record: ConvFinQARecord,
    answer_list: list[str],
    reasoning_log: list[str],
    document_context: str,
    width: int = 80,
    log_file=None,
) -> None:
    """Print a detailed inspection of the agent's response for a single record"""

    def log(msg: str = "") -> None:
        if log_file:
            log_file.write(msg + "\n")
        else:
            logger.info(msg)

    def wrap(text: str, prefix: str) -> str:
        return textwrap.fill(
            text,
            width=width,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )

    log("=" * width)
    log("Document Context:")
    log(document_context)

    for i, (question, answer, reasoning, gt, program) in enumerate(
        zip(
            record.dialogue.conv_questions,
            answer_list,
            reasoning_log,
            record.dialogue.executed_answers,
            record.dialogue.turn_program,
            strict=True,
        )
    ):
        correct = evaluate_question(answer, str(gt))
        log("=" * width)
        log(wrap(question, f"Question {i + 1}: "))
        log(wrap(reasoning, "Solver Reasoning: "))
        log(wrap(answer, "AI:         "))
        log(f"GT:         {gt}  {'✓' if correct else '✗'}")
        log(f"GT Program: {program}")


def build_evaluation_dataframe(
    records, responses, answers_list, reasoning_logs, record_times, results
):
    """Build a DataFrame of the evaluation results for all records with detailed info."""
    df_result = pd.DataFrame(results).reset_index()
    df_result["question"] = [r.dialogue.conv_questions for r in records]
    df_result["ground_truths"] = [r.dialogue.executed_answers for r in records]
    df_result["ai_answer"] = answers_list
    df_result["solver_reasoning"] = reasoning_logs
    df_result["record_time_s"] = record_times
    df_result["context"] = [r["document_context"] for r in responses]
    return df_result


def compute_turn_level_accuracy(
    df_result: pd.DataFrame,
    max_turns: int | None = 6,
    turn_flags_col: str = "turn_flags",
) -> pd.Series:
    """Return mean accuracy (%) per turn position (1-indexed)."""
    turn_df = pd.DataFrame(df_result[turn_flags_col].tolist())

    if max_turns is not None:
        turn_df = turn_df.iloc[:, :max_turns]

    turn_df.columns = np.arange(1, len(turn_df.columns) + 1)
    return turn_df.mean(skipna=True) * 100


def plot_turn_level_accuracy(
    df_result: pd.DataFrame,
    max_turns: int | None = 6,
    turn_flags_col: str = "turn_flags",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of per-turn accuracy matching ConvFinQA Figure 5."""
    turn_accuracy = compute_turn_level_accuracy(
        df_result=df_result,
        max_turns=max_turns,
        turn_flags_col=turn_flags_col,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(turn_accuracy))
    bars = ax.bar(x, turn_accuracy.values, color="#4472C4", label="Execution Accuracy")

    for bar, val in zip(bars, turn_accuracy.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4472C4",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(turn_accuracy.index)
    ax.set_xlabel("$n$-th conversation turn")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(turn_accuracy.values) * 1.2)
    ax.legend()
    ax.set_title("Performances for the $n$th conversation turn")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


# ------------------ Run and track experiment & artifacts with MLFlow -----------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("convfinqa-experiment")
mlflow.langchain.autolog(disable=True, silent=True)

run_name = agent_runtime["run_name"]
source_file = agent_runtime["source_file"]

# Selected the first 50 records for evaluation to balance between runtime and statistical significance
# Can be increased and randomly sampled for more robust evaluation if needed
records = get_records("dev")[:50]

with mlflow.start_run(run_name=run_name, description=""):
    # Log parameters and input files to MLFlow
    mlflow.log_artifact(source_file, artifact_path="agent_code")
    mlflow.log_params({"solver_" + k: v for k, v in asdict(solver_config).items()})
    mlflow.log_text(prompts["solver"], "solver_prompt_template.txt")

    if reflector_config is not None:
        mlflow.log_params(
            {"reflector_" + k: v for k, v in asdict(reflector_config).items()}
        )
        mlflow.log_text(prompts["reflector"], "reflector_prompt_template.txt")

    # Run the agent on all selected records and evaluate results
    start_time = time()
    responses, answers, reasoning_logs, record_times = zip(
        *asyncio.run(run_records_async(records, max_concurrency=30)), strict=True
    )
    end_time = time()

    results = [
        evaluate_single_record(record, ans)
        for record, ans in zip(records, answers, strict=True)
    ]
    df_result = build_evaluation_dataframe(
        records, responses, answers, reasoning_logs, record_times, results
    )

    overall_accuracy = df_result["turn_flags"].explode().mean()
    avg_record_time_s = float(np.mean(record_times))
    logger.info("overall_accuracy: %.6f", overall_accuracy)
    logger.info("avg_record_time_s: %.3f", avg_record_time_s)

    turn_accuracy = compute_turn_level_accuracy(df_result, max_turns=6)
    fig = plot_turn_level_accuracy(df_result, max_turns=5)

    # Log overall metrics, turn-level accuracy, and artifacts to MLFlow
    mlflow.log_metric("execution_time_mins", (end_time - start_time) / 60)
    mlflow.log_metric("execution_accuracy", overall_accuracy)
    mlflow.log_metrics(
        {
            f"turn_{int(turn)}_accuracy": float(acc)
            for turn, acc in turn_accuracy.items()
        }
    )
    mlflow.log_metric("avg_record_time_s", avg_record_time_s)
    mlflow.log_table(df_result, "evaluation_results.json")
    mlflow.log_figure(fig, "turn_level_accuracy.png")
    plt.close(fig)

    # Log inspection logs to a temporary directory and upload as artifacts to MLFlow
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (record, response, ans, reas_log) in enumerate(
            zip(records, responses, answers, reasoning_logs, strict=True)
        ):
            record_id = record.id.replace("/", "_")
            record_path = os.path.join(tmp_dir, f"record_{i:02d}_{record_id}.txt")
            with open(record_path, "w") as record_file:
                inspect_response(
                    record,
                    ans,
                    reas_log,
                    response["document_context"],
                    log_file=record_file,
                )

        mlflow.log_artifacts(tmp_dir, artifact_path="record_inspections")


# ------------------------------------------------------------------
# Test the agent on single record for debugging and inspection (uncomment to run)
# record_id = "Single_IPG/2009/page_89.pdf-3"
# record_id = "Single_ETR/2004/page_213.pdf-4"
# record_id = "Single_BLK/2014/page_119.pdf-4"  # last q: portion
# record_id = "Double_STT/2008/page_83.pdf"
# record_id = "Single_DVN/2007/page_58.pdf-3"

# record = get_record(record_id)
# response, answers, reasoning_log = run_single_record(record)
# inspect_response(record, answers, reasoning_log, response["document_context"], 100)
# eval_result = evaluate_single_record(record, answers)
