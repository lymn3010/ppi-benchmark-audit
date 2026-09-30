"""Single construction point for parser backends."""
from __future__ import annotations

from src.parsing.backend import ParserBackend


def build_parser_backend(
    name: str = "stanza",
    *,
    stanza_package: str = "craft",
    use_coref: bool = False,
    use_nominalization: bool = False,
    qanom_threshold: float = 0.5,
    use_gpu: bool = True,
) -> ParserBackend:
    backend = (name or "stanza").lower()
    if backend == "stanza":
        from src.parsing.stanza import StanzaAdapter

        return StanzaAdapter(
            package=stanza_package,
            use_coref=use_coref,
            use_nominalization=use_nominalization,
            qanom_threshold=qanom_threshold,
            use_gpu=use_gpu,
        )
    raise ValueError(f"Unsupported parser backend {name!r}")
