"""Tests for the RiskScorer."""

from __future__ import annotations

import pytest

from aushadhi.models import (
    EvidenceLevel,
    Interaction,
    InteractionMechanism,
    Severity,
)
from aushadhi.risk_scorer import RiskScorer


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


def _make_interaction(
    severity: Severity = Severity.MODERATE,
    mechanism: InteractionMechanism = InteractionMechanism.CYP450_INHIBITION,
    evidence: EvidenceLevel = EvidenceLevel.ESTABLISHED,
    drugs: list[str] | None = None,
) -> Interaction:
    return Interaction(
        drugs=drugs or ["drug_a", "drug_b"],
        severity=severity,
        mechanism=mechanism,
        evidence=evidence,
        description="Test interaction",
        clinical_significance="Test significance",
        management="Test management",
    )


class TestRiskScoring:
    def test_contraindicated_gets_max_score(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.CONTRAINDICATED)
        assessment = scorer.score(ix)
        assert assessment.risk_score == 10.0

    def test_severe_scores_high(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.SEVERE)
        assessment = scorer.score(ix)
        assert assessment.risk_score >= 7.0

    def test_moderate_scores_medium(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.MODERATE)
        assessment = scorer.score(ix)
        assert 3.0 <= assessment.risk_score <= 7.0

    def test_mild_scores_low(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.MILD)
        assessment = scorer.score(ix)
        assert assessment.risk_score <= 4.0

    def test_theoretical_evidence_reduces_score(self, scorer: RiskScorer) -> None:
        established = _make_interaction(evidence=EvidenceLevel.ESTABLISHED)
        theoretical = _make_interaction(evidence=EvidenceLevel.THEORETICAL)
        score_est = scorer.score(established).risk_score
        score_theo = scorer.score(theoretical).risk_score
        assert score_est > score_theo

    def test_qt_mechanism_adds_bonus(self, scorer: RiskScorer) -> None:
        qt = _make_interaction(mechanism=InteractionMechanism.QT_PROLONGATION)
        generic = _make_interaction(mechanism=InteractionMechanism.OTHER)
        score_qt = scorer.score(qt).risk_score
        score_gen = scorer.score(generic).risk_score
        assert score_qt > score_gen

    def test_higher_order_gets_complexity_bonus(self, scorer: RiskScorer) -> None:
        pairwise = _make_interaction(drugs=["a", "b"])
        triple = _make_interaction(drugs=["a", "b", "c"])
        score_pair = scorer.score(pairwise).risk_score
        score_triple = scorer.score(triple).risk_score
        assert score_triple > score_pair

    def test_score_clamped_to_10(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(
            severity=Severity.CONTRAINDICATED,
            mechanism=InteractionMechanism.QT_PROLONGATION,
            drugs=["a", "b", "c", "d", "e"],
        )
        assessment = scorer.score(ix)
        assert assessment.risk_score <= 10.0


class TestRiskAssessmentContent:
    def test_assessment_has_summary(self, scorer: RiskScorer) -> None:
        ix = _make_interaction()
        assessment = scorer.score(ix)
        assert assessment.summary
        assert "drug_a" in assessment.summary or "drug_b" in assessment.summary

    def test_assessment_has_recommendations(self, scorer: RiskScorer) -> None:
        ix = _make_interaction()
        assessment = scorer.score(ix)
        assert len(assessment.recommendations) >= 1

    def test_contraindicated_recommendation_says_avoid(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.CONTRAINDICATED)
        assessment = scorer.score(ix)
        recs_text = " ".join(assessment.recommendations).lower()
        assert "avoid" in recs_text

    def test_assessment_has_monitoring_parameters(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(mechanism=InteractionMechanism.BLEEDING_RISK)
        assessment = scorer.score(ix)
        assert len(assessment.monitoring_parameters) >= 1


class TestAggregateRisk:
    def test_no_interactions_returns_zero(self, scorer: RiskScorer) -> None:
        assert scorer.aggregate_risk([]) == 0.0

    def test_single_interaction(self, scorer: RiskScorer) -> None:
        ix = _make_interaction(severity=Severity.SEVERE)
        assessments = scorer.score_all([ix])
        agg = scorer.aggregate_risk(assessments)
        assert agg == assessments[0].risk_score

    def test_multiple_interactions_increase_aggregate(self, scorer: RiskScorer) -> None:
        interactions = [
            _make_interaction(severity=Severity.MODERATE, drugs=["a", "b"]),
            _make_interaction(severity=Severity.MODERATE, drugs=["b", "c"]),
            _make_interaction(severity=Severity.MODERATE, drugs=["a", "c"]),
        ]
        assessments = scorer.score_all(interactions)
        agg = scorer.aggregate_risk(assessments)
        single = assessments[0].risk_score
        assert agg > single

    def test_aggregate_capped_at_10(self, scorer: RiskScorer) -> None:
        interactions = [
            _make_interaction(severity=Severity.CONTRAINDICATED, drugs=["a", "b"]),
            _make_interaction(severity=Severity.CONTRAINDICATED, drugs=["b", "c"]),
            _make_interaction(severity=Severity.CONTRAINDICATED, drugs=["a", "c"]),
            _make_interaction(severity=Severity.SEVERE, drugs=["a", "d"]),
        ]
        assessments = scorer.score_all(interactions)
        agg = scorer.aggregate_risk(assessments)
        assert agg <= 10.0

    def test_score_all_sorted_descending(self, scorer: RiskScorer) -> None:
        interactions = [
            _make_interaction(severity=Severity.MILD),
            _make_interaction(severity=Severity.SEVERE),
            _make_interaction(severity=Severity.MODERATE),
        ]
        assessments = scorer.score_all(interactions)
        scores = [a.risk_score for a in assessments]
        assert scores == sorted(scores, reverse=True)
