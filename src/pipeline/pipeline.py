"""Reader pipeline: StanzaAdapter -> ParsedSentence -> events -> identity -> ProjectionPlanner."""
from __future__ import annotations

import logging
from typing import Optional

from .contracts import EntryData
from .protein_targets import ProteinTargets
from .record_builder import RecordBuilder

from src.models import Record
from src.runtime_resources import RuntimeResources, load_runtime_resources
from src.downstream.projection_planner import ProjectionPlanner
from src.downstream.reference_policy import CanonicalIdentityMap
from src.downstream.parsed_reference_resolver import ParsedReferenceResolver
from src.extraction.parsers.dep.dep_parser import SemanticEventExtractor
from src.models.candidate import CandidateRelationLedger
from src.utils.colors import RED, RESET
from src.parsing import ParsedSentence, ParserBackend


class Pipeline:
    def __init__(
        self,
        parser: ParserBackend,
        resources: RuntimeResources | None = None,
        parser_name: str = "stanza",
        *,
        mine_lexicalized_paths: bool = False,
        suppress_self_alias_pairs: bool = False,
        emit_event_observations: bool = False,
        debug_errors: bool = False,
    ) -> None:
        # Preserve the selected backend name in run provenance.
        self.parser_name = parser_name
        self.resources = resources or load_runtime_resources()
        # Optional lexicalized-path audit rows (off by default).
        self.mine_lexicalized_paths = mine_lexicalized_paths
        self.suppress_self_alias_pairs = suppress_self_alias_pairs
        self.emit_event_observations = emit_event_observations
        self.debug_errors = debug_errors

        # Every backend returns the parser-neutral boundary directly.
        self.parser = parser

        # Parser-neutral realization channels emit CandidateEvents.
        self.event_extractor = SemanticEventExtractor(resources=self.resources)
        self.parsed_reference_resolver = ParsedReferenceResolver(resources=self.resources)

        # Shared semantic interpretation: canonical identity then pair projection.
        self.candidate_relation_ledger = CandidateRelationLedger()
        self.projection_planner = ProjectionPlanner(resources=self.resources)

    EntryData = EntryData

    # Compatibility helpers; implementations belong to the named stage.
    _normalize_parsed_targets = staticmethod(ProteinTargets._normalize_parsed_targets)
    _protein_entities_from_parsed = staticmethod(ProteinTargets._protein_entities_from_parsed)
    _self_alias_candidate_pairs_from_parsed = staticmethod(ProteinTargets._self_alias_candidate_pairs_from_parsed)
    _suppress_self_alias_pairs_n = staticmethod(ProteinTargets._suppress_self_alias_pairs_n)
    _protein_canonical_ids_n = staticmethod(ProteinTargets._protein_canonical_ids_n)
    _protein_names_n = staticmethod(ProteinTargets._protein_names_n)
    _canonical_protein_name = staticmethod(ProteinTargets._canonical_protein_name)
    _is_self_alias_pair = staticmethod(ProteinTargets._is_self_alias_pair)
    _all_projected_pairs = staticmethod(RecordBuilder._all_projected_pairs)
    _project_event_observations = staticmethod(RecordBuilder._project_event_observations)
    _project_mention_graph = staticmethod(RecordBuilder._project_mention_graph)
    _project_pair_inference_traces = staticmethod(RecordBuilder._project_pair_inference_traces)
    _annotate_candidate_relation_decisions = staticmethod(RecordBuilder._annotate_candidate_relation_decisions)
    _normalize_pair_value = staticmethod(RecordBuilder._normalize_pair_value)
    _normalize_pair_values = staticmethod(RecordBuilder._normalize_pair_values)

    def process_sentence(
        self,
        sentence: str,
        entry_data: Optional[EntryData] = None,
        verbose: bool = False,
    ) -> Optional[Record]:
        entry_data = {
            "id": "",
            "text": sentence,
            "gold_pairs": None,
        } if entry_data is None else entry_data

        try:
            result = self.parser.parse(sentence, entry_data["id"])
        except Exception as e:
            if self.debug_errors:
                raise
            logging.error(f"[Parser] {entry_data['id']} | {e} | {sentence}")
            print(f"\n{RED}[ERROR]{RESET} {e} ({entry_data['id']}: {sentence})")
            return None

        if not isinstance(result, ParsedSentence):
            raise TypeError(
                f"{type(self.parser).__name__}.parse() must return ParsedSentence; "
                f"got {type(result).__name__}"
            )
        return self.process_parsed(result, entry_data, verbose)

    # Parser-neutral ParsedSentence path

    def process_parsed(
        self,
        parsed: ParsedSentence,
        entry_data: Optional[EntryData] = None,
        verbose: bool = False,
    ) -> Optional[Record]:
        """Run event extraction and projection on a ParsedSentence."""
        entry_data = {
            "id": parsed.sentence_id,
            "text": parsed.text,
            "gold_pairs": None,
        } if entry_data is None else entry_data

        # Reference propagation must use normalized dataset protein indices.
        parsed = self._normalize_parsed_targets(parsed, entry_data)

        parsed = self.parsed_reference_resolver.resolve(parsed)

        current_stage = "Init"
        try:
            current_stage = "SemanticEventExtractor"
            candidates = self.event_extractor.extract(parsed, entry_data["id"])

            if verbose:
                print(
                    f" {'[' + parsed.parser_name.upper() + '] ' + str(len(candidates)) + ' candidate event(s)'}"
                    .center(80, "-")
                )

            current_stage = "CandidateRelationLedger"
            candidate_relations = self.candidate_relation_ledger.build(
                candidates,
                sentence_id=entry_data["id"],
            )
            current_stage = "CanonicalIdentityMap"
            canonical_identity_map = CanonicalIdentityMap.from_candidate_events(candidates)
            current_stage = "ProjectionPlanner"
            projection_plans = self.projection_planner.plan_all(
                candidates,
                sentence_id=entry_data["id"],
                identity_map=canonical_identity_map,
            )
            self.projection_planner.apply_plans(candidates, projection_plans)

            suppressed_pairs = []
            if self.suppress_self_alias_pairs:
                suppressed_pairs = self._suppress_self_alias_pairs_n(candidates, parsed, entry_data)

            if verbose:
                for event in candidates:
                    print(
                        f"[{event.event_id}] {event.extraction_detail}: "
                        f"{list(event.projected_pairs)}"
                    )

            return RecordBuilder(
                self.parser, self.resources,
                emit_event_observations=self.emit_event_observations,
                mine_lexicalized_paths=self.mine_lexicalized_paths,
            ).build(
                parsed,
                entry_data,
                candidates,
                parser_backend=parsed.parser_name,
                candidate_relations=candidate_relations,
                projection_plans=projection_plans,
                canonical_identity_map=canonical_identity_map,
                suppressed_pairs=suppressed_pairs,
            )

        except Exception as e:
            if self.debug_errors:
                raise
            logging.error(f"[{current_stage}] {entry_data['id']} | {e} | {parsed.text}")
            print(f"\n{RED}[ERROR]{RESET} {e} ({entry_data['id']}: {parsed.text})")
            return None


    # Parser-neutral protein normalization and coref propagation


def init_default_pipeline(
    use_coref_resolver: bool = True,
    use_nominal_detector: bool = True,
    parser=None,
    resources: RuntimeResources | None = None,
    parser_name: str = "stanza",
    mine_lexicalized_paths: bool = False,
    suppress_self_alias_pairs: bool = False,
    emit_event_observations: bool = False,
    # Native-Stanza options.
    stanza_package: str = "craft",
    qanom_threshold: float = 0.5,
    use_gpu: bool = True,
    debug_errors: bool = False,
) -> Pipeline:
    """Initialize the extraction pipeline from a parser-neutral backend."""
    if parser is None:
        from src.parsing import build_parser_backend
        parser = build_parser_backend(
            parser_name,
            stanza_package=stanza_package,
            use_coref=use_coref_resolver,
            use_nominalization=use_nominal_detector,
            qanom_threshold=qanom_threshold,
            use_gpu=use_gpu,
        )

    return Pipeline(
        parser,
        resources or load_runtime_resources(),
        parser_name=parser_name,
        mine_lexicalized_paths=mine_lexicalized_paths,
        suppress_self_alias_pairs=suppress_self_alias_pairs,
        emit_event_observations=emit_event_observations,
        debug_errors=debug_errors,
    )
