from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RunConfig:
    """Shared configuration for all pipeline modes."""

    dataset: str = "aimed"
    split: str = "train"
    limit: int = -1
    # Sampled runs are recorded separately from full-corpus results.
    sample_size: int | None = None
    sample_seed: int = 42

    use_coref: bool = True
    use_nominalization: bool = True
    use_parser_cache: bool = True
    use_gpu: bool = True

    # Parser backend. Stanza CRAFT is the only supported production parser.
    parser_name: str = "stanza"
    stanza_package: str = "craft"
    # QANom classifier confidence threshold.
    qanom_threshold: float = 0.5
    suppress_self_alias_pairs: bool = False

    # Review evidence is recorded separately from pair projection.
    mine_lexicalized_paths: bool = True

    emit_event_observations: bool = True

    # Ignore the run registry cache.
    force_rerun: bool = False
