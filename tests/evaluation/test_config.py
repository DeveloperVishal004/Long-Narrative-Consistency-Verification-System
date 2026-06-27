"""AblationVariant / EvaluationConfig validation and fingerprint tests."""

from lncvs.evaluation import AblationVariant, EvaluationConfig, FusionStrategy, standard_ablation_matrix


def test_ablation_variant_defaults_to_full_pipeline() -> None:
    variant = AblationVariant(name="full")
    assert variant.use_question_generation is True
    assert variant.use_bm25 is True
    assert variant.fusion_strategy is FusionStrategy.RRF


def test_ablation_variant_fingerprint_is_stable_for_identical_settings() -> None:
    variant_a = AblationVariant(name="full")
    variant_b = AblationVariant(name="full")
    assert variant_a.fingerprint() == variant_b.fingerprint()


def test_ablation_variant_fingerprint_ignores_name() -> None:
    """Two variants with identical settings but different display names are
    the same point in the ablation space and must fingerprint identically."""
    variant_a = AblationVariant(name="full")
    variant_b = AblationVariant(name="baseline")
    assert variant_a.fingerprint() == variant_b.fingerprint()


def test_ablation_variant_fingerprint_differs_for_different_settings() -> None:
    variant_a = AblationVariant(name="full")
    variant_b = AblationVariant(name="no_bm25", use_bm25=False)
    assert variant_a.fingerprint() != variant_b.fingerprint()


def test_standard_ablation_matrix_has_four_variants_including_full() -> None:
    variants = standard_ablation_matrix()
    names = {variant.name for variant in variants}
    assert names == {"full", "no_question_generation", "no_bm25", "no_rrf"}


def test_standard_ablation_matrix_each_variant_toggles_exactly_one_component() -> None:
    variants = {variant.name: variant for variant in standard_ablation_matrix()}

    assert variants["no_question_generation"].use_question_generation is False
    assert variants["no_question_generation"].use_bm25 is True
    assert variants["no_question_generation"].fusion_strategy is FusionStrategy.RRF

    assert variants["no_bm25"].use_bm25 is False
    assert variants["no_bm25"].use_question_generation is True

    assert variants["no_rrf"].fusion_strategy is FusionStrategy.ROUND_ROBIN
    assert variants["no_rrf"].use_bm25 is True
    assert variants["no_rrf"].use_question_generation is True


def test_evaluation_config_defaults() -> None:
    config = EvaluationConfig()
    assert config.k_cutoffs == [5, 10]
    assert config.seed == 0
    assert config.persist_ledgers is False


def test_evaluation_config_fingerprint_is_stable_and_differs_on_change() -> None:
    config_a = EvaluationConfig(seed=0)
    config_b = EvaluationConfig(seed=0)
    config_c = EvaluationConfig(seed=1)

    assert config_a.fingerprint() == config_b.fingerprint()
    assert config_a.fingerprint() != config_c.fingerprint()
