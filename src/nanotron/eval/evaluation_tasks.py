# [MODULE DOC]
# Purpose: LightEval custom-task definitions used by nanotron's evaluation runner (passed to LightEval
# as a custom tasks file). Exposes TASKS_TABLE, the list of LightevalTaskConfig objects LightEval loads.
# Contents: TASKS_TABLE starts with LightEval's multilingual task table, then adds English tasks, each
# in three formulations (MCF multiple-choice letters, CF cloze, Hybrid) named
# "<task>_<formulation>[:subset]" in suite "custom" with pinned dataset revisions:
# ARC Easy/Challenge (test split, train few-shots), HellaSwag (validation), CommonsenseQA,
# OpenBookQA, Winogrande xl (continuation prompts; fixed metric list with no/token/char normalization
# regardless of formulation), PIQA, MMLU for each of the 57 subjects in MMLU_SUBSETS (test split, dev
# few-shots) and MMLU-Pro (test split, validation few-shots). Metrics are log-likelihood accuracy with
# token, character and (for some tasks) PMI normalization, adapted to the formulation by
# get_metrics_for_formulation. Lambdas adapt each dataset row to question/choices/gold index.
# Import-time side effects: builds all task configs (imports lighteval). As a script, prints the
# task names and count.
# KNOWN ISSUE: the script branch prints a generator object rather than the task names.
# KNOWN ISSUE: Winogrande uses a metric list that ignores the formulation, unlike the other tasks.
from functools import partial

from lighteval.metrics.dynamic_metrics import (
    loglikelihood_acc_metric,
)
from lighteval.metrics.normalizations import LogProbCharNorm, LogProbPMINorm, LogProbTokenNorm
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.adapters import (
    winogrand_adapter,
)
from lighteval.tasks.multilingual.tasks import TASKS_TABLE as ML_TASKS_TABLE
from lighteval.tasks.multilingual.utils.task_utils import get_metrics_for_formulation
from lighteval.tasks.templates.continuation import get_continuation_prompt_function
from lighteval.tasks.templates.hellaswag import get_hellaswag_prompt_function
from lighteval.tasks.templates.multichoice import get_mcq_prompt_function
from lighteval.tasks.templates.utils.formulation import (
    CFFormulation,
    HybridFormulation,
    MCFFormulation,
)
from lighteval.utils.language import Language

TASKS_TABLE = []

TASKS_TABLE.extend(ML_TASKS_TABLE)

arc_tasks = [
    LightevalTaskConfig(
        name=f"arc_{formulation.name.lower()}:{subset.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="allenai/ai2_arc",
        hf_subset=f"ARC-{subset}",
        hf_revision="210d026faf9955653af8916fad021475a3f00453",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="train",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in ["Easy", "Challenge"]
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(arc_tasks)

hellaswag_tasks = [
    LightevalTaskConfig(
        name=f"hellaswag_{formulation.name.lower()}",
        suite=["custom"],
        prompt_function=get_hellaswag_prompt_function(
            language=Language.ENGLISH,
            adapter=lambda line: {
                "activity_label": line["activity_label"],
                "ctx_a": line["ctx_a"],
                "ctx_b": line["ctx_b"],
                "continuations": line["endings"],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        hf_repo="Rowan/hellaswag",
        hf_subset="default",
        hf_revision="6002345709e0801764318f06bf06ce1e7d1a1fe3",
        evaluation_splits=["validation"],
        hf_avail_splits=["validation"],
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
        trust_dataset=True,
    )
    for formulation in [MCFFormulation(), CFFormulation(), HybridFormulation()]
]

TASKS_TABLE.extend(hellaswag_tasks)

commonsense_qa_tasks = [
    LightevalTaskConfig(
        name=f"commonsenseqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": line["choices"]["label"].index(line["answerKey"].strip()),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="tau/commonsense_qa",
        hf_subset="default",
        hf_revision="94630fe30dad47192a8546eb75f094926d47e155",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(commonsense_qa_tasks)

openbook_qa_tasks = [
    LightevalTaskConfig(
        name=f"openbookqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question_stem"],
                "choices": line["choices"]["text"],
                "gold_idx": LETTER_INDICES.index(line["answerKey"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="allenai/openbookqa",
        hf_subset="main",
        hf_revision="388097ea7776314e93a529163e0fea805b8a6454",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(openbook_qa_tasks)

winogrande_tasks = [
    LightevalTaskConfig(
        name=f"winogrande_{formulation.name.lower()}",
        suite=("custom",),
        prompt_function=get_continuation_prompt_function(
            Language.ENGLISH, partial(winogrand_adapter, Language.ENGLISH), formulation=formulation
        ),
        hf_repo="allenai/winogrande",
        hf_subset="winogrande_xl",
        trust_dataset=True,
        hf_revision="85ac5b5a3b7a930e22d590176e39460400d19e41",
        metric=[
            loglikelihood_acc_metric(normalization=None),
            loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
            loglikelihood_acc_metric(normalization=LogProbCharNorm()),
        ],
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(winogrande_tasks)

piqa_tasks = [
    LightevalTaskConfig(
        name=f"piqa_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["goal"],
                "choices": [line["sol1"], line["sol2"]],
                "gold_idx": int(line["label"]),
            },
            formulation=formulation,
        ),
        suite=["custom"],
        hf_repo="ybisk/piqa",
        hf_revision="2e8ac2dffd59bac8c3c6714948f4c551a0848bb0",
        hf_subset="plain_text",
        trust_dataset=True,
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(piqa_tasks)


MMLU_SUBSETS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

mmlu_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_{formulation.name.lower()}:{subset}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": int(line["answer"]),
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="cais/mmlu",
        hf_subset=subset,
        hf_revision="c30699e8356da336a370243923dbaf21066bb9fe",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="dev",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for subset in MMLU_SUBSETS
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(mmlu_tasks)

mmlu_pro_tasks = [
    LightevalTaskConfig(
        name=f"mmlu_pro_{formulation.name.lower()}",
        prompt_function=get_mcq_prompt_function(
            Language.ENGLISH,
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["answer_index"],
            },
            formulation=formulation,
        ),
        suite=("custom",),
        hf_repo="TIGER-Lab/MMLU-Pro",
        hf_subset="default",
        hf_revision="3373e0b32277875b8db2aa555a333b78a08477ea",
        trust_dataset=True,
        evaluation_splits=("test",),
        few_shots_split="validation",
        metric=get_metrics_for_formulation(
            formulation,
            [
                loglikelihood_acc_metric(normalization=LogProbTokenNorm()),
                loglikelihood_acc_metric(normalization=LogProbCharNorm()),
                loglikelihood_acc_metric(normalization=LogProbPMINorm()),
            ],
        ),
    )
    for formulation in [
        MCFFormulation(),
        CFFormulation(),
        HybridFormulation(),
    ]
]

TASKS_TABLE.extend(mmlu_pro_tasks)


if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
