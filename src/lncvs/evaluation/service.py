"""EvaluationHarness: drives a LedgerProducer across a dataset and produces
lightweight, fingerprinted EvaluationReports.

Typed against lncvs.evaluation.runner.LedgerProducer (a Protocol), not the
concrete PipelineRunner -- this module never imports PipelineRunner or
orchestration.LangGraphPipeline. Either satisfies the Protocol structurally;
callers (tests, CLI) wire a concrete runner in, so EvaluationHarness never
needs a hard import of either implementation. This is also why this module
never imports orchestration/: doing so would create a cycle the moment
orchestration/ needs an evaluation/-only type (see schemas/evaluation.py's
module docstring for the AblationVariant relocation this avoids).

Never embeds a full EvidenceLedger inside a report -- only its fingerprint
and the metrics computed from it (see schemas/evaluation.py's module
docstring for the dependency-direction reasoning behind this). Ledgers and
reports are persisted to disk only as an explicit opt-in: see
EvaluationConfig.persist_ledgers below.
"""

import hashlib
from pathlib import Path

from lncvs.chunking import chunk_document
from lncvs.evaluation.config import EvaluationConfig
from lncvs.evaluation.dataset import map_spans_to_chunks
from lncvs.evaluation.fingerprint import ledger_fingerprint
from lncvs.evaluation.metrics.citation import compute_citation_metrics
from lncvs.evaluation.metrics.latency import compute_latency_metrics
from lncvs.evaluation.metrics.retrieval import compute_retrieval_metrics
from lncvs.evaluation.metrics.verdict import compute_verdict_metrics
from lncvs.evaluation.reporting import save_ledger, save_report
from lncvs.evaluation.runner import LedgerProducer
from lncvs.ingestion import load_and_clean_narrative
from lncvs.schemas import AblationVariant, VerdictEnum
from lncvs.schemas.evaluation import (
    AblatedComponent,
    AblationReport,
    CitationMetrics,
    ContributionDelta,
    EvaluationDataset,
    EvaluationReport,
    ExampleResult,
    LatencyMetrics,
    ProvenanceFingerprints,
    RankCutoffMetric,
    RetrievalMetrics,
)

_ABLATED_COMPONENT_VARIANT_NAMES = {
    AblatedComponent.QUESTION_GENERATION: "no_question_generation",
    AblatedComponent.BM25: "no_bm25",
    AblatedComponent.RRF: "no_rrf",
}


class EvaluationHarness:
    """Drives a LedgerProducer over a gold dataset, for one or many AblationVariants."""

    def __init__(self, runner: LedgerProducer, config: EvaluationConfig) -> None:
        self._runner = runner
        self._config = config

    def evaluate_variant(self, dataset: EvaluationDataset, variant: AblationVariant) -> EvaluationReport:
        """Run every example in dataset under variant, returning one aggregated EvaluationReport."""
        example_results: list[ExampleResult] = []
        verdict_pairs: list[tuple[VerdictEnum, VerdictEnum]] = []
        retrieval_metrics_list: list[RetrievalMetrics] = []
        citation_metrics_list: list[CitationMetrics] = []

        for example in dataset.examples:
            narrative_path = Path(example.narrative_path)
            # source_id must match PipelineRunner.run()'s convention exactly:
            # chunk_id is a hash of (source_id, char_start, char_end, text), so
            # a different source_id here would silently produce gold_chunk_ids
            # that can never match the chunk_ids the runner actually indexed.
            document = load_and_clean_narrative(narrative_path, source_id=str(narrative_path))
            chunks = chunk_document(document, self._runner.chunking_config)
            gold_chunk_ids = map_spans_to_chunks(example.gold_evidence, chunks)

            ledger = self._runner.run(narrative_path, example.original_claim, variant)

            if ledger.final_verdict is None:
                raise ValueError(f"PipelineRunner produced a ledger with no final_verdict for example {example.example_id!r}")
            predicted_verdict = ledger.final_verdict.verdict
            verdict_pairs.append((example.expected_verdict, predicted_verdict))

            retrieval_metrics = compute_retrieval_metrics(ledger, gold_chunk_ids, self._config.k_cutoffs)
            citation_metrics = compute_citation_metrics(ledger, gold_chunk_ids)
            latency_metrics = compute_latency_metrics(ledger)

            if retrieval_metrics is not None:
                retrieval_metrics_list.append(retrieval_metrics)
            if citation_metrics is not None:
                citation_metrics_list.append(citation_metrics)

            fingerprint = ledger_fingerprint(ledger)
            ledger_path: str | None = None
            if self._config.persist_ledgers:
                ledger_filename = f"{example.example_id}_{variant.name}_{fingerprint}"
                saved_path = save_ledger(ledger, Path(self._config.output_dir) / "ledgers", ledger_filename)
                ledger_path = str(saved_path)

            example_results.append(
                ExampleResult(
                    example_id=example.example_id,
                    predicted_verdict=predicted_verdict,
                    expected_verdict=example.expected_verdict,
                    fired_rule=ledger.final_verdict.fired_rule,
                    correct=(predicted_verdict == example.expected_verdict),
                    retrieval=retrieval_metrics,
                    citation=citation_metrics,
                    latency=latency_metrics,
                    ledger_fingerprint=fingerprint,
                    ledger_path=ledger_path,
                )
            )

        verdict_metrics = compute_verdict_metrics(verdict_pairs)
        aggregate_retrieval = _average_retrieval(retrieval_metrics_list, self._config.k_cutoffs)
        aggregate_citation = _average_citation(citation_metrics_list)
        aggregate_latency = _average_latency(example_results)

        dataset_fp = dataset.fingerprint()
        variant_fp = variant.fingerprint()
        eval_config_fp = self._config.fingerprint()
        provenance = ProvenanceFingerprints(eval_config_fp=eval_config_fp, seed=self._config.seed)
        run_id = hashlib.sha256(f"{dataset_fp}:{variant_fp}:{eval_config_fp}".encode("utf-8")).hexdigest()[:16]

        report = EvaluationReport(
            run_id=run_id,
            variant_name=variant.name,
            variant_fingerprint=variant_fp,
            dataset_id=dataset.dataset_id,
            dataset_fingerprint=dataset_fp,
            provenance=provenance,
            verdict=verdict_metrics,
            retrieval=aggregate_retrieval,
            citation=aggregate_citation,
            latency=aggregate_latency,
            example_results=example_results,
            example_count=len(example_results),
        )

        if self._config.persist_ledgers:
            save_report(report, Path(self._config.output_dir))

        return report

    def run_ablation(self, dataset: EvaluationDataset, variants: list[AblationVariant]) -> AblationReport:
        """Evaluate every variant and compute each ablated component's contribution
        to verdict accuracy, relative to the "full" variant. variants must include
        a variant named "full" and at least one of the standard ablated variants."""
        if not any(variant.name == "full" for variant in variants):
            raise ValueError('run_ablation requires one variant named "full" to compute contribution deltas against')

        reports = [self.evaluate_variant(dataset, variant) for variant in variants]
        full_report = next(report for report in reports if report.variant_name == "full")

        deltas: list[ContributionDelta] = []
        for component, variant_name in _ABLATED_COMPONENT_VARIANT_NAMES.items():
            ablated_report = next((report for report in reports if report.variant_name == variant_name), None)
            if ablated_report is None:
                continue
            with_value = full_report.verdict.accuracy
            without_value = ablated_report.verdict.accuracy
            deltas.append(
                ContributionDelta(
                    component=component,
                    metric_name="verdict_accuracy",
                    with_value=with_value,
                    without_value=without_value,
                    delta=with_value - without_value,
                )
            )

        return AblationReport(reports=reports, deltas=deltas)


def _average_retrieval(metrics_list: list[RetrievalMetrics], k_cutoffs: list[int]) -> RetrievalMetrics | None:
    if not metrics_list:
        return None

    mrr = sum(metrics.mrr for metrics in metrics_list) / len(metrics_list)
    cutoffs: list[RankCutoffMetric] = []
    for k in sorted(k_cutoffs):
        matching = [cutoff for metrics in metrics_list for cutoff in metrics.cutoffs if cutoff.k == k]
        if not matching:
            continue
        recall = sum(cutoff.recall for cutoff in matching) / len(matching)
        precision = sum(cutoff.precision for cutoff in matching) / len(matching)
        cutoffs.append(RankCutoffMetric(k=k, recall=recall, precision=precision))

    if not cutoffs:
        return None
    return RetrievalMetrics(mrr=mrr, cutoffs=cutoffs)


def _average_citation(metrics_list: list[CitationMetrics]) -> CitationMetrics | None:
    if not metrics_list:
        return None

    total_cited = sum(metrics.cited_count for metrics in metrics_list)
    total_grounded = sum(metrics.grounded_count for metrics in metrics_list)
    return CitationMetrics(
        citation_accuracy=total_grounded / total_cited if total_cited else 0.0,
        hallucination_rate=(total_cited - total_grounded) / total_cited if total_cited else 0.0,
        cited_count=total_cited,
        grounded_count=total_grounded,
    )


def _average_latency(example_results: list[ExampleResult]) -> LatencyMetrics:
    if not example_results:
        return LatencyMetrics(stages=[], end_to_end_ms=0.0)

    end_to_end_ms = sum(result.latency.end_to_end_ms for result in example_results) / len(example_results)
    return LatencyMetrics(stages=[], end_to_end_ms=end_to_end_ms)
