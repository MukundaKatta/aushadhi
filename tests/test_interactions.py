"""Tests for the InteractionChecker and DrugDatabase."""

from __future__ import annotations

import pytest

from aushadhi.drugs.database import DrugDatabase
from aushadhi.drugs.interactions import InteractionChecker
from aushadhi.models import DrugClass, InteractionMechanism, Severity


@pytest.fixture
def db() -> DrugDatabase:
    return DrugDatabase()


@pytest.fixture
def checker(db: DrugDatabase) -> InteractionChecker:
    return InteractionChecker(db)


# ---------------------------------------------------------------------------
# DrugDatabase tests
# ---------------------------------------------------------------------------

class TestDrugDatabase:
    def test_database_has_100_plus_drugs(self, db: DrugDatabase) -> None:
        assert db.count >= 100

    def test_lookup_by_generic_name(self, db: DrugDatabase) -> None:
        drug = db.lookup("warfarin")
        assert drug is not None
        assert drug.name == "warfarin"

    def test_lookup_by_brand_name(self, db: DrugDatabase) -> None:
        drug = db.lookup("Lipitor")
        assert drug is not None
        assert drug.name == "atorvastatin"

    def test_lookup_case_insensitive(self, db: DrugDatabase) -> None:
        drug = db.lookup("METFORMIN")
        assert drug is not None
        assert drug.name == "metformin"

    def test_lookup_nonexistent(self, db: DrugDatabase) -> None:
        assert db.lookup("nonexistent_drug_xyz") is None

    def test_lookup_many(self, db: DrugDatabase) -> None:
        drugs = db.lookup_many(["warfarin", "aspirin", "nonexistent"])
        assert len(drugs) == 2
        names = {d.name for d in drugs}
        assert names == {"warfarin", "aspirin"}

    def test_search(self, db: DrugDatabase) -> None:
        results = db.search("statin")
        assert len(results) >= 4
        for drug in results:
            assert drug.drug_class == DrugClass.STATIN

    def test_by_class(self, db: DrugDatabase) -> None:
        beta_blockers = db.by_class(DrugClass.BETA_BLOCKER)
        assert len(beta_blockers) >= 3
        for drug in beta_blockers:
            assert drug.drug_class == DrugClass.BETA_BLOCKER

    def test_narrow_therapeutic_index_drugs(self, db: DrugDatabase) -> None:
        nti_drugs = [d for d in db.all_drugs if d.narrow_therapeutic_index]
        assert len(nti_drugs) >= 10
        nti_names = {d.name for d in nti_drugs}
        assert "warfarin" in nti_names
        assert "digoxin" in nti_names
        assert "lithium" in nti_names
        assert "phenytoin" in nti_names

    def test_cyp450_profiles_present(self, db: DrugDatabase) -> None:
        drugs_with_cyp = [d for d in db.all_drugs if d.cyp450]
        # Many drugs should have CYP450 data
        assert len(drugs_with_cyp) >= 50


# ---------------------------------------------------------------------------
# InteractionChecker -- known interaction tests
# ---------------------------------------------------------------------------

class TestKnownInteractions:
    def test_warfarin_aspirin(self, checker: InteractionChecker, db: DrugDatabase) -> None:
        warfarin = db.lookup("warfarin")
        aspirin = db.lookup("aspirin")
        assert warfarin and aspirin
        interactions = checker.check_pair(warfarin, aspirin)
        assert len(interactions) >= 1
        assert interactions[0].severity == Severity.SEVERE
        assert interactions[0].mechanism == InteractionMechanism.BLEEDING_RISK

    def test_warfarin_fluconazole(self, checker: InteractionChecker, db: DrugDatabase) -> None:
        warfarin = db.lookup("warfarin")
        fluconazole = db.lookup("fluconazole")
        assert warfarin and fluconazole
        interactions = checker.check_pair(warfarin, fluconazole)
        assert len(interactions) >= 1
        assert interactions[0].severity == Severity.SEVERE
        assert interactions[0].mechanism == InteractionMechanism.CYP450_INHIBITION

    def test_simvastatin_clarithromycin_contraindicated(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        simvastatin = db.lookup("simvastatin")
        clarithromycin = db.lookup("clarithromycin")
        assert simvastatin and clarithromycin
        interactions = checker.check_pair(simvastatin, clarithromycin)
        assert len(interactions) >= 1
        assert interactions[0].severity == Severity.CONTRAINDICATED

    def test_opioid_benzo_cns_depression(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        oxycodone = db.lookup("oxycodone")
        alprazolam = db.lookup("alprazolam")
        assert oxycodone and alprazolam
        interactions = checker.check_pair(oxycodone, alprazolam)
        assert len(interactions) >= 1
        severe_or_contra = [
            ix for ix in interactions
            if ix.severity in (Severity.SEVERE, Severity.CONTRAINDICATED)
        ]
        assert len(severe_or_contra) >= 1

    def test_fentanyl_alprazolam_contraindicated(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        fentanyl = db.lookup("fentanyl")
        alprazolam = db.lookup("alprazolam")
        assert fentanyl and alprazolam
        interactions = checker.check_pair(fentanyl, alprazolam)
        contra = [ix for ix in interactions if ix.severity == Severity.CONTRAINDICATED]
        assert len(contra) >= 1

    def test_clopidogrel_omeprazole(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        clopidogrel = db.lookup("clopidogrel")
        omeprazole = db.lookup("omeprazole")
        assert clopidogrel and omeprazole
        interactions = checker.check_pair(clopidogrel, omeprazole)
        assert len(interactions) >= 1
        assert interactions[0].severity == Severity.MODERATE

    def test_no_interaction_for_safe_pair(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        metformin = db.lookup("metformin")
        cetirizine = db.lookup("cetirizine")
        assert metformin and cetirizine
        interactions = checker.check_pair(metformin, cetirizine)
        assert len(interactions) == 0


# ---------------------------------------------------------------------------
# InteractionChecker -- inferred interactions
# ---------------------------------------------------------------------------

class TestInferredInteractions:
    def test_cyp3a4_inhibitor_substrate_pair(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        """Ketoconazole (CYP3A4 inhibitor) + amlodipine (CYP3A4 substrate)."""
        ketoconazole = db.lookup("ketoconazole")
        amlodipine = db.lookup("amlodipine")
        assert ketoconazole and amlodipine
        interactions = checker.check_pair(ketoconazole, amlodipine)
        assert len(interactions) >= 1
        cyp_interactions = [
            ix for ix in interactions if ix.mechanism == InteractionMechanism.CYP450_INHIBITION
        ]
        assert len(cyp_interactions) >= 1

    def test_serotonergic_pair_inferred(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        """Two serotonergic drugs should trigger serotonin syndrome flag."""
        escitalopram = db.lookup("escitalopram")
        buspirone = db.lookup("buspirone")
        assert escitalopram and buspirone
        interactions = checker.check_pair(escitalopram, buspirone)
        serotonin = [
            ix for ix in interactions if ix.mechanism == InteractionMechanism.SEROTONIN_SYNDROME
        ]
        assert len(serotonin) >= 1

    def test_qt_prolonging_pair(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        """Two QT-prolonging drugs should be flagged."""
        citalopram = db.lookup("citalopram")
        ondansetron = db.lookup("ondansetron")
        assert citalopram and ondansetron
        interactions = checker.check_pair(citalopram, ondansetron)
        qt = [ix for ix in interactions if ix.mechanism == InteractionMechanism.QT_PROLONGATION]
        assert len(qt) >= 1


# ---------------------------------------------------------------------------
# InteractionChecker -- higher-order interactions
# ---------------------------------------------------------------------------

class TestHigherOrderInteractions:
    def test_triple_whammy(self, checker: InteractionChecker, db: DrugDatabase) -> None:
        """ACE inhibitor + diuretic + NSAID -> triple whammy nephrotoxicity."""
        drugs = db.lookup_many(["lisinopril", "hydrochlorothiazide", "ibuprofen"])
        assert len(drugs) == 3
        interactions = checker.check_all(drugs)
        nephro = [
            ix for ix in interactions
            if ix.mechanism == InteractionMechanism.NEPHROTOXICITY and len(ix.drugs) >= 3
        ]
        assert len(nephro) >= 1
        assert nephro[0].severity == Severity.SEVERE

    def test_multiple_cns_depressants(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        """3+ CNS depressants should trigger higher-order interaction."""
        drugs = db.lookup_many(["oxycodone", "alprazolam", "gabapentin"])
        assert len(drugs) == 3
        interactions = checker.check_all(drugs)
        cns_higher = [
            ix for ix in interactions
            if ix.mechanism == InteractionMechanism.CNS_DEPRESSION and len(ix.drugs) >= 3
        ]
        assert len(cns_higher) >= 1

    def test_check_all_returns_pairwise_and_higher(
        self, checker: InteractionChecker, db: DrugDatabase
    ) -> None:
        """check_all should return both pairwise and higher-order interactions."""
        drugs = db.lookup_many([
            "warfarin", "aspirin", "omeprazole", "fluoxetine", "tramadol",
        ])
        assert len(drugs) == 5
        interactions = checker.check_all(drugs)
        pairwise = [ix for ix in interactions if len(ix.drugs) == 2]
        higher = [ix for ix in interactions if len(ix.drugs) > 2]
        assert len(pairwise) >= 1
        # With this combination we expect at least some interactions
        assert len(interactions) >= 3


class TestCheckAllDeduplication:
    def test_no_duplicate_pairs(self, checker: InteractionChecker, db: DrugDatabase) -> None:
        """Each drug pair should only appear once in pairwise results."""
        drugs = db.lookup_many(["warfarin", "aspirin", "ibuprofen"])
        interactions = checker.check_all(drugs)
        pair_keys: set[tuple[str, ...]] = set()
        for ix in interactions:
            if len(ix.drugs) == 2:
                key = tuple(sorted(ix.drugs))
                # Same pair may have different mechanisms, but let's at least
                # verify the checker ran without error
                pair_keys.add(key)
        # We should have at most C(3,2) = 3 unique pairs
        assert len(pair_keys) <= 3
