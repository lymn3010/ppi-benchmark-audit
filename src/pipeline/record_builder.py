"""Assemble saved reader evidence; audit side channels never feed inference."""
from __future__ import annotations

import logging
from src.parsing import ParsedSentence
from src.models import Record
from src.models.candidate import CandidateEvent, CandidateRelation
from src.downstream.reference_policy import CanonicalIdentityMap
from .contracts import EntryData

from .protein_targets import ProteinTargets


class RecordBuilder:
    def __init__(self, parser, resources, *, emit_event_observations=False, mine_lexicalized_paths=False):
        self.parser = parser
        self.resources = resources
        self.emit_event_observations = emit_event_observations
        self.mine_lexicalized_paths = mine_lexicalized_paths

    def build(
        self,
        parsed: ParsedSentence,
        entry_data: EntryData,
        candidate_events: list[CandidateEvent],
        *,
        parser_backend: str,
        candidate_relations: list[CandidateRelation] | None = None,
        projection_plans: list | None = None,
        canonical_identity_map: CanonicalIdentityMap | None = None,
        suppressed_pairs: list[dict] | None = None,
    ) -> Record:
        """Build a Record from a ParsedSentence."""
        token_offsets = [(n.char_start, n.char_end) for n in parsed.nodes]
        target_count = len(parsed.protein_map)

        provenance = {
            "sentence_id": entry_data["id"],
            "sentence_text": entry_data["text"],
            "target_count": target_count,
            "parser_backend": parser_backend,
        }
        candidate_rows = [row.to_dict() for row in (candidate_relations or [])]
        candidate_event_rows = [c.to_dict() for c in (candidate_events or [])]
        plan_rows = [plan.to_dict() for plan in (projection_plans or [])]
        self._annotate_candidate_relation_decisions(
            candidate_rows,
            plan_rows,
        )

        parse_artifacts: dict = {
            "parser_backend": parser_backend,
            "token_offsets": [list(pair) for pair in token_offsets],
        }
        from src.analysis.syntax_tree_serialization import serialize_parsed_trees
        parse_artifacts.update(
            serialize_parsed_trees(parsed, candidate_events=candidate_event_rows)
        )
        artifact_provider = getattr(self.parser, "parse_artifacts", None)
        if artifact_provider is not None:
            parse_artifacts.update(
                artifact_provider(parsed, candidate_events=candidate_event_rows) or {}
            )

        # Drop timing/batching fields from cached parser evidence.
        raw_parser_data = getattr(parsed, "raw_parser_data", None)
        if raw_parser_data:
            parse_artifacts["stanza_raw"] = {
                key: value
                for key, value in raw_parser_data.items()
                if key not in {"parse_ms", "nominalization_ms", "batched"}
            }

        artifacts: dict = {
            "parse": parse_artifacts,
            "runtime_rule_toggles": self.resources.rule_toggle_manifest(),
            "candidate_relations": candidate_rows,
            "candidate_events": candidate_event_rows,
            "identity_equivalences": list(
                (canonical_identity_map or CanonicalIdentityMap()).declarations
            ),
            "protein_entities": ProteinTargets._protein_entities_from_parsed(parsed, entry_data),
            "self_alias_candidate_pairs": ProteinTargets._self_alias_candidate_pairs_from_parsed(
                parsed, entry_data
            ),
        }
        if suppressed_pairs:
            artifacts["suppressed_pairs"] = suppressed_pairs
        if getattr(self, "emit_event_observations", False):
            artifacts["mention_graph"] = self._project_mention_graph(parsed)
            artifacts["event_observations"] = self._project_event_observations(
                candidate_events or [],
                parsed,
            )
            artifacts["projection_plans"] = [
                plan.to_dict() for plan in (projection_plans or [])
            ]
            artifacts["canonical_identity_map"] = (
                canonical_identity_map.to_dict()
                if canonical_identity_map is not None else CanonicalIdentityMap().to_dict()
            )
            artifacts["pair_inference_traces"] = self._project_pair_inference_traces(
                sentence_id=entry_data["id"],
                candidate_events=candidate_event_rows,
                candidate_relations=candidate_rows,
                projection_plans=plan_rows,
                predicted_pairs=self._all_projected_pairs(candidate_events),
                gold_pairs=entry_data["gold_pairs"],
                parser_ok=bool(token_offsets),
            )

        if self.mine_lexicalized_paths:
            artifacts["path_candidates"] = self._mine_paths_from_parsed(
                parsed=parsed,
                entry_data=entry_data,
                predicted_pairs=self._all_projected_pairs(candidate_events),
                candidate_rows=candidate_rows,
            )

        return Record(
            sentence_id=entry_data["id"],
            sentence_text=entry_data["text"],
            target_count=target_count,
            events=candidate_events,
            predicted_pairs=self._all_projected_pairs(candidate_events),
            gold_pairs=entry_data["gold_pairs"],
            token_offsets=token_offsets,
            provenance=provenance,
            artifacts=artifacts,
        )


    @staticmethod
    def _all_projected_pairs(events: list[CandidateEvent]) -> list[tuple[int, int]]:
        return sorted({pair for event in events for pair in event.projected_pairs})


    @staticmethod
    def _project_event_observations(
        candidate_events: list[CandidateEvent],
        parsed: ParsedSentence | None = None,
    ) -> list[dict]:
        """Build the additive observation side channel without affecting inference."""
        from src.models.observation import project_observation

        return [project_observation(event, parsed).to_dict() for event in candidate_events]


    @staticmethod
    def _project_mention_graph(parsed: ParsedSentence) -> dict:
        """Build the lossless parser-fact graph before semantic interpretation."""
        from src.models.observation import build_mention_graph

        return build_mention_graph(parsed).to_dict()


    @staticmethod
    def _project_pair_inference_traces(**kwargs) -> list[dict]:
        """Build honest pair traces without changing extraction or inference."""
        from src.analysis.inference_trace import build_pair_inference_traces

        return [trace.to_dict() for trace in build_pair_inference_traces(**kwargs)]


    def _mine_paths_from_parsed(
        self,
        *,
        parsed: ParsedSentence,
        entry_data: EntryData,
        predicted_pairs: list[tuple[int, int]],
        candidate_rows: list[dict],
    ) -> list[dict]:
        try:
            from src.analysis.lexicalized_path_miner import mine_path_candidates_from_parsed
            return mine_path_candidates_from_parsed(
                parsed=parsed,
                sentence_id=entry_data["id"],
                gold_pairs=entry_data["gold_pairs"],
                predicted_pairs=predicted_pairs,
                candidate_relation_rows=candidate_rows,
            )
        except Exception as exc:
            logging.debug("[PathMiner] %s", exc)
            return []


    @staticmethod
    def _annotate_candidate_relation_decisions(
        candidate_rows: list[dict],
        plan_rows: list[dict],
    ) -> None:
        """Mark which candidate relation rows became emitted pairs."""
        plans_by_event = {
            str(row.get("source_event_id") or ""): row
            for row in plan_rows or []
            if row.get("source_event_id")
        }
        for row in candidate_rows:
            source_event_id = str(row.get("source_event_id") or "")
            plan = plans_by_event.get(source_event_id)
            if not plan:
                row.setdefault("projection_status", "")
                row.setdefault("projection_reason_code", "")
                row.setdefault("projection_plan_id", "")
                row.setdefault("projection_pair_selected", False)
                continue

            final_pairs = RecordBuilder._normalize_pair_values(
                plan.get("final_pairs") or plan.get("computed_pairs") or []
            )
            pair = RecordBuilder._normalize_pair_value(row.get("pair"))
            selected = pair is not None and pair in final_pairs
            projection_status = str(plan.get("application_status") or "")

            row["projection_reason_code"] = str(plan.get("reason_code") or "")
            row["projection_plan_id"] = str(plan.get("id") or "")
            row["projection_pair_selected"] = bool(selected)

            if selected:
                row["projection_status"] = projection_status
                row["decision"] = "applied"
            elif projection_status == "review_only":
                row["projection_status"] = "review_only"
                row["decision"] = "review_only"
            elif projection_status == "not_applied":
                row["projection_status"] = "not_applied"
                row["decision"] = "not_applied"
            elif projection_status == "applied":
                row["projection_status"] = "not_selected"
                row["decision"] = "not_selected"
            else:
                row["projection_status"] = projection_status


    @staticmethod
    def _normalize_pair_value(value) -> tuple[int, int] | None:
        if value is None or len(value) != 2:
            return None
        a, b = int(value[0]), int(value[1])
        if a == b:
            return None
        return tuple(sorted((a, b)))


    @staticmethod
    def _normalize_pair_values(values) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for value in values or []:
            pair = RecordBuilder._normalize_pair_value(value)
            if pair is not None:
                out.add(pair)
        return out
