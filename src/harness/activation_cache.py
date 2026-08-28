"""Experiment 12 / Epic 04 (small scale) — teacher activation capture.

Runs the real ``EleutherAI/pythia-14m`` teacher over a bundled text corpus and
captures the **inputs** of selected linear projections with forward pre-hooks.
From each capture it produces:

- a fitting second moment ``C = X_fit^T X_fit / N`` (the only statistic the
  closed-form activation-aware fits need), and
- a held-out evaluation activation matrix used to measure on-distribution
  layer output error — never the same samples the fit saw.

Statistics are cached on disk (``.cache/activations/``) so repeated runs are
deterministic and cheap. Nothing captured here is inference-time model state:
second moments exist only at fitting time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

TEACHER_REPO_ID: str = "EleutherAI/pythia-14m"
DEFAULT_CACHE_DIRECTORY: Path = Path(".cache/activations")
DEFAULT_SEQUENCE_LENGTH: int = 128
DEFAULT_EVALUATION_FRACTION: float = 0.2
DEFAULT_MAX_EVALUATION_COLUMNS: int = 1024

CAPTURED_PROJECTIONS: tuple[str, ...] = (
    "gpt_neox.layers.0.attention.query_key_value",
    "gpt_neox.layers.0.mlp.dense_4h_to_h",
    "gpt_neox.layers.3.attention.query_key_value",
    "gpt_neox.layers.3.mlp.dense_4h_to_h",
)
"""Attention QKV and MLP down projections from an early and a middle layer —
the two roles the program-wide advancement gate requires."""

CALIBRATION_CORPUS: tuple[str, ...] = (
    "The quick brown fox jumps over the lazy dog while the farmer watches "
    "from the porch, wondering whether the harvest will survive the frost.",
    "In 1969, engineers guided a spacecraft to the lunar surface using "
    "computers less powerful than a modern wristwatch, relying on careful "
    "checklists and redundant systems.",
    "The recipe calls for two cups of flour, a teaspoon of baking soda, "
    "three eggs, and a pinch of salt, folded together until the batter is "
    "smooth but not overworked.",
    "Quarterly revenue rose by twelve percent, driven primarily by strong "
    "demand in the enterprise segment, although margins compressed due to "
    "elevated component costs.",
    "She walked along the riverbank at dusk, listening to the water slide "
    "over the stones, and thought about the letter she had never sent.",
    "The theorem states that every continuous function on a closed and "
    "bounded interval attains both a maximum and a minimum value.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n - 1) + "
    "fibonacci(n - 2)  # classic recursive definition with exponential cost",
    "Parliament debated the measure for three days before a narrow vote "
    "sent the amended bill back to committee for further revision.",
    "Photosynthesis converts carbon dioxide and water into glucose and "
    "oxygen, using energy captured from sunlight by chlorophyll molecules.",
    "The orchestra tuned to the oboe's A, the conductor raised the baton, "
    "and the hall fell into the particular silence that precedes music.",
    "Storm warnings were issued for the coastal counties as the hurricane "
    "strengthened overnight, with landfall expected before dawn on Tuesday.",
    "He repaired the old clock with a screwdriver, a drop of oil, and more "
    "patience than he knew he had, and it kept fair time for years after.",
    "The museum's new wing houses artifacts recovered from the shipwreck, "
    "including navigational instruments, coins, and a remarkably preserved "
    "leather-bound logbook.",
    "A hash table offers average constant-time lookup by mapping keys "
    "through a hash function into an array of buckets, resolving collisions "
    "by chaining or open addressing.",
    "Grandmother's garden overflowed with tomatoes and basil by late July, "
    "and the kitchen smelled of simmering sauce every Sunday afternoon.",
    "The court held that the statute, as applied, violated the petitioner's "
    "rights, and remanded the case for proceedings consistent with the "
    "opinion.",
    "Migration patterns of the arctic tern span both hemispheres, with some "
    "individuals flying more than seventy thousand kilometers in a single "
    "year.",
    "Mix the epoxy in small batches, clamp the joint firmly, and allow "
    "twenty-four hours of cure time before sanding flush with the grain.",
    "The startup pivoted twice before finding product-market fit, and the "
    "founders later credited their survival to a stubbornly small burn rate.",
    "Rain fell steadily on the tin roof of the field station while the "
    "researchers logged soil samples and argued cheerfully about dinner.",
    "Interest rates influence borrowing costs, asset prices, and exchange "
    "rates, which is why central bank announcements move markets within "
    "seconds.",
    "The chess engine sacrificed its queen on move nineteen, and the "
    "grandmaster stared at the board for eleven minutes before resigning.",
    "Volcanic ash grounded flights across the continent for nearly a week, "
    "stranding travelers and disrupting supply chains from florists to "
    "factories.",
    "To install the package, create a virtual environment, activate it, and "
    "run pip install with the requirements file pinned to exact versions.",
    "The lighthouse keeper kept a journal of every ship that passed, every "
    "storm that broke, and every quiet morning when the sea lay flat as "
    "glass.",
    "Enzymes lower the activation energy of biochemical reactions by "
    "stabilizing transition states, which is why a cell at body temperature "
    "can perform chemistry that would otherwise require a furnace.",
    "The negotiators worked through the night, trading concessions on "
    "tariffs for guarantees on inspection schedules, and announced a "
    "framework agreement shortly before the markets opened.",
    "He learned to sail on a battered dinghy with a patched mainsail, "
    "capsizing twice a week until the wind stopped feeling like an enemy "
    "and started feeling like a conversation.",
    "A binary search tree degenerates into a linked list when keys arrive "
    "in sorted order, which is why balanced variants rotate nodes to keep "
    "the height logarithmic in the number of elements.",
    "The glacier retreated four hundred meters in a decade, leaving behind "
    "a moraine of crushed rock and a meltwater lake that had not existed "
    "when the oldest villagers were children.",
    "Season the cast-iron skillet by rubbing it with a thin film of oil and "
    "baking it upside down for an hour, repeating until the surface turns "
    "glossy black and sheds water like a lotus leaf.",
    "The violinist practiced the passage at half tempo for a week, then at "
    "three-quarter tempo for another, and by the concert the sixteenth "
    "notes sounded inevitable rather than difficult.",
    "Supply outpaced demand for the third consecutive quarter, forcing the "
    "refinery to cut throughput and idle two of its five distillation "
    "columns until inventories normalized.",
    "In the folktale, the miller's youngest daughter outwits the river "
    "spirit three times, and each victory costs her something she does not "
    "learn to miss until winter.",
    "The immune system distinguishes self from non-self through a training "
    "process in the thymus, where lymphocytes that bind too strongly to "
    "the body's own proteins are eliminated.",
    "City engineers rerouted the streetcar line around the sinkhole, "
    "welded temporary plates over the exposed utilities, and promised a "
    "permanent repair before the first snowfall.",
    "She catalogued the beetle specimens by wing casing and antenna "
    "segment, correcting three misidentifications that had stood in the "
    "collection since the previous century.",
    "The compiler inlines small functions, unrolls short loops, and hoists "
    "invariant computations out of hot paths, transformations that "
    "preserve semantics while reshaping the instruction stream.",
    "Monsoon rains arrived two weeks late, and the farmers who had staked "
    "their season on early planting watched the sky with an arithmetic of "
    "worry known to every dryland generation.",
    "The auction house verified the painting's provenance through customs "
    "stamps, gallery ledgers, and a photograph of the artist's studio in "
    "which the unfinished canvas leans against a chair.",
    "Bees communicate the direction and distance of forage through a "
    "waggle dance performed on the vertical comb, encoding the angle to "
    "the sun in the orientation of the run.",
    "The submarine's inertial navigation drifted a few meters per hour, so "
    "the crew periodically raised a mast to fix their position against "
    "satellites before slipping back beneath the thermocline.",
    "Grandfather kept the ledger in pencil because ink, he said, was for "
    "people who never made mistakes, and the eraser smudges told the true "
    "history of the store.",
    "A well-designed API makes the easy things easy and the hard things "
    "possible, names its concepts consistently, and fails loudly at the "
    "boundary rather than quietly in the interior.",
    "The marathon's final miles climbed through the old quarter, past "
    "balconies of strangers ringing cowbells, and she counted lampposts "
    "because the finish line was still an abstraction.",
    "Continental crust rides higher than oceanic crust because it is "
    "thicker and less dense, a buoyancy argument that explains why the "
    "planet has both abyssal plains and high plateaus.",
    "The bakery's sourdough starter survived two moves, one power outage, "
    "and a summer of neglect, fed back to vigor each time with nothing "
    "more than flour, water, and patience.",
    "Regulators required the bank to hold additional capital against its "
    "trading book after stress tests revealed concentrated exposure to "
    "commercial real estate in three metropolitan markets.",
    "The observatory scheduled the survey for moonless nights, stacking "
    "hundreds of short exposures to tease faint galaxies out of a sky "
    "brighter than the signal itself.",
    "He whittled the whistle from a willow shoot the way his aunt had "
    "shown him, tapping the bark loose in one piece and cutting the "
    "window before sliding the sleeve back on.",
)


@dataclass(frozen=True)
class ActivationStatistics:
    """Cached activation statistics for one captured projection input.

    Attributes:
        projection_name: Full module path of the captured linear projection.
        second_moment: (dim, dim) float32 fitting second moment
            ``E[x x^T]`` over the fitting split.
        evaluation_activations: (dim, n_eval) float32 held-out activation
            columns for on-distribution output-error measurement.
        fit_sample_count: Token positions in the fitting split.
    """

    projection_name: str
    second_moment: torch.Tensor
    evaluation_activations: torch.Tensor
    fit_sample_count: int

    def __post_init__(self) -> None:
        dimension = self.second_moment.shape[0]
        if self.second_moment.shape != (dimension, dimension):
            raise ValueError(
                f"second_moment must be square, got "
                f"{tuple(self.second_moment.shape)}"
            )
        if self.evaluation_activations.shape[0] != dimension:
            raise ValueError(
                "evaluation_activations rows "
                f"{self.evaluation_activations.shape[0]} != second_moment "
                f"dimension {dimension}"
            )
        if self.fit_sample_count <= 0:
            raise ValueError(
                f"fit_sample_count must be positive, got {self.fit_sample_count}"
            )


def split_captured_activations(
    captured_activations: torch.Tensor,
    projection_name: str,
    evaluation_fraction: float = DEFAULT_EVALUATION_FRACTION,
    max_evaluation_columns: int = DEFAULT_MAX_EVALUATION_COLUMNS,
    seed: int = 0,
) -> ActivationStatistics:
    """Deterministically split captured samples and build statistics.

    Args:
        captured_activations: (n_samples, dim) float32 activation rows.
        projection_name: Module path the samples were captured from.
        evaluation_fraction: Fraction of samples held out for evaluation.
        max_evaluation_columns: Cap on held-out evaluation columns.
        seed: Permutation seed for the deterministic split.

    Returns:
        Fitting second moment plus held-out evaluation activations.
    """
    if captured_activations.dim() != 2:
        raise ValueError(
            "captured_activations must be 2-D (samples, dim), got "
            f"{tuple(captured_activations.shape)}"
        )
    sample_count = captured_activations.shape[0]
    if sample_count < 4:
        raise ValueError(
            f"need at least 4 captured samples to split, got {sample_count}"
        )
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError(
            f"evaluation_fraction must be in (0, 1), got {evaluation_fraction}"
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(sample_count, generator=generator)
    evaluation_count = min(
        max(1, int(round(evaluation_fraction * sample_count))),
        max_evaluation_columns,
    )
    evaluation_rows = captured_activations[permutation[:evaluation_count]]
    fitting_rows = captured_activations[permutation[evaluation_count:]]

    fitting_rows_64 = fitting_rows.to(torch.float64)
    second_moment = (
        fitting_rows_64.T @ fitting_rows_64 / fitting_rows_64.shape[0]
    ).to(torch.float32)

    return ActivationStatistics(
        projection_name=projection_name,
        second_moment=second_moment,
        evaluation_activations=evaluation_rows.T.contiguous().to(torch.float32),
        fit_sample_count=int(fitting_rows.shape[0]),
    )


def capture_pythia_activation_statistics(
    projection_names: tuple[str, ...] = CAPTURED_PROJECTIONS,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    cache_directory: Path = DEFAULT_CACHE_DIRECTORY,
    seed: int = 0,
) -> dict[str, ActivationStatistics]:
    """Capture (or load cached) real teacher activation statistics.

    Runs ``EleutherAI/pythia-14m`` over the bundled corpus once, capturing
    the inputs of each requested projection with forward pre-hooks, then
    splits fit/eval deterministically. Results are cached to
    ``cache_directory`` keyed by projection name and corpus geometry.

    Returns:
        Mapping from projection name to its :class:`ActivationStatistics`.
    """
    cache_directory.mkdir(parents=True, exist_ok=True)
    corpus_fingerprint = f"c{len(CALIBRATION_CORPUS)}"
    cache_key = (
        f"{TEACHER_REPO_ID.replace('/', '__')}_{corpus_fingerprint}"
        f"_seq{sequence_length}_seed{seed}"
    )

    statistics: dict[str, ActivationStatistics] = {}
    missing_projections = []
    for projection_name in projection_names:
        cache_path = cache_directory / f"{cache_key}__{projection_name}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, weights_only=True)
            statistics[projection_name] = ActivationStatistics(
                projection_name=projection_name,
                second_moment=payload["second_moment"],
                evaluation_activations=payload["evaluation_activations"],
                fit_sample_count=int(payload["fit_sample_count"]),
            )
        else:
            missing_projections.append(projection_name)

    if not missing_projections:
        return statistics

    captured = _run_teacher_and_capture(
        tuple(missing_projections), sequence_length
    )
    for projection_name, activation_rows in captured.items():
        projection_statistics = split_captured_activations(
            activation_rows, projection_name, seed=seed
        )
        statistics[projection_name] = projection_statistics
        cache_path = cache_directory / f"{cache_key}__{projection_name}.pt"
        torch.save(
            {
                "second_moment": projection_statistics.second_moment,
                "evaluation_activations": (
                    projection_statistics.evaluation_activations
                ),
                "fit_sample_count": projection_statistics.fit_sample_count,
            },
            cache_path,
        )
    return statistics


def _run_teacher_and_capture(
    projection_names: tuple[str, ...], sequence_length: int
) -> dict[str, torch.Tensor]:
    """Forward the bundled corpus through the teacher, capturing inputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_REPO_ID, torch_dtype=torch.float32
    )
    model.eval()

    token_ids = tokenizer(
        "\n\n".join(CALIBRATION_CORPUS), return_tensors="pt"
    ).input_ids[0]
    usable_length = (token_ids.shape[0] // sequence_length) * sequence_length
    if usable_length == 0:
        raise RuntimeError(
            f"corpus tokenizes to {token_ids.shape[0]} tokens, fewer than one "
            f"sequence of length {sequence_length}"
        )
    batched_input_ids = token_ids[:usable_length].reshape(-1, sequence_length)

    module_map = dict(model.named_modules())
    captured_rows: dict[str, list[torch.Tensor]] = {
        name: [] for name in projection_names
    }
    hook_handles = []

    def make_hook(projection_name: str):
        def capture_input(_module, inputs):
            hidden_states = inputs[0].detach().to(torch.float32)
            captured_rows[projection_name].append(
                hidden_states.reshape(-1, hidden_states.shape[-1])
            )

        return capture_input

    try:
        for projection_name in projection_names:
            if projection_name not in module_map:
                raise KeyError(
                    f"projection {projection_name!r} not found in "
                    f"{TEACHER_REPO_ID}"
                )
            hook_handles.append(
                module_map[projection_name].register_forward_pre_hook(
                    make_hook(projection_name)
                )
            )
        with torch.no_grad():
            model(batched_input_ids)
    finally:
        for handle in hook_handles:
            handle.remove()

    return {
        name: torch.cat(rows, dim=0) for name, rows in captured_rows.items()
    }


def load_projection_weight(projection_name: str) -> torch.Tensor:
    """Load one projection's dense weight matrix from the teacher checkpoint."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    checkpoint_path = hf_hub_download(TEACHER_REPO_ID, "model.safetensors")
    state = load_file(checkpoint_path)
    return state[f"{projection_name}.weight"].to(torch.float32)
