use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{
    GlweSecretKeyEntity, LweSecretKeyEntity, LweSeededBootstrapKeyEntity,
};
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    LweSeededBootstrapKeyCreationError for LweSeededBootstrapKeyCreationEngine @
    NullDecompositionBaseLog => "The key decomposition base log must be greater than zero.",
    NullDecompositionLevelCount => "The key decomposition level count must be greater than zero.",
    DecompositionTooLarge => "The decomposition precision (base log * level count) must not exceed \
                              the precision of the ciphertext."
}

impl<EngineError: std::error::Error> LweSeededBootstrapKeyCreationError<EngineError> {
    pub fn perform_generic_checks(
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        integer_precision: usize,
    ) -> Result<(), Self> {
        if decomposition_base_log.0 == 0 {
            return Err(Self::NullDecompositionBaseLog);
        }
        if decomposition_level_count.0 == 0 {
            return Err(Self::NullDecompositionLevelCount);
        }
        if decomposition_base_log.0 * decomposition_level_count.0 > integer_precision {
            return Err(Self::DecompositionTooLarge);
        }
        Ok(())
    }
}

/// A trait for engines creating seeded LWE bootstrap keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation creates an LWE bootstrap key from the
/// `input_key` LWE secret key, and the `output_key` GLWE secret key.
///
/// # Formal Definition
///
/// cf [`here`](`crate::specification::entities::LweSeededBootstrapKeyEntity`)
pub trait LweSeededBootstrapKeyCreationEngine<LweSecretKey, GlweSecretKey, SeededBootstrapKey>:
    AbstractEngine
where
    SeededBootstrapKey: LweSeededBootstrapKeyEntity,
    LweSecretKey: LweSecretKeyEntity<KeyDistribution = SeededBootstrapKey::InputKeyDistribution>,
    GlweSecretKey: GlweSecretKeyEntity<KeyDistribution = SeededBootstrapKey::OutputKeyDistribution>,
{
    /// Creates a seeded LWE bootstrap key.
    fn create_lwe_seeded_bootstrap_key(
        &mut self,
        input_key: &LweSecretKey,
        output_key: &GlweSecretKey,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> Result<SeededBootstrapKey, LweSeededBootstrapKeyCreationError<Self::EngineError>>;

    /// Unsafely creates a seeded LWE bootstrap key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweSeededBootstrapKeyCreationError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.
    unsafe fn create_lwe_seeded_bootstrap_key_unchecked(
        &mut self,
        input_key: &LweSecretKey,
        output_key: &GlweSecretKey,
        decomposition_base_log: DecompositionBaseLog,
        decomposition_level_count: DecompositionLevelCount,
        noise: Variance,
    ) -> SeededBootstrapKey;
}