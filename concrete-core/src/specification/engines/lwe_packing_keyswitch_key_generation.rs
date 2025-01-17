use super::engine_error;
use crate::prelude::{GlweSecretKeyEntity, LwePackingKeyswitchKeyEntity};
use crate::specification::engines::AbstractEngine;

use crate::specification::entities::LweSecretKeyEntity;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

engine_error! {
    LwePackingKeyswitchKeyGenerationError for LwePackingKeyswitchKeyGenerationEngine @
    NullDecompositionBaseLog => "The key decomposition base log must be greater than zero.",
    NullDecompositionLevelCount => "The key decomposition level count must be greater than zero.",
    DecompositionTooLarge => "The decomposition precision (base log * level count) must not exceed \
                              the precision of the ciphertext."
}

impl<EngineError: std::error::Error> LwePackingKeyswitchKeyGenerationError<EngineError> {
    /// Validates the inputs
    pub fn perform_generic_checks(
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        integer_precision: usize,
    ) -> Result<(), Self> {
        if decomposition_base_log.0 == 0 {
            return Err(Self::NullDecompositionBaseLog);
        }

        if decomposition_level_count.0 == 0 {
            return Err(Self::NullDecompositionLevelCount);
        }

        if decomposition_level_count.0 * decomposition_base_log.0 > integer_precision {
            return Err(Self::DecompositionTooLarge);
        }

        Ok(())
    }
}

/// A trait for engines generating new LWE packing keyswitch keys.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a new LWE packing keyswitch key
/// allowing to switch from the `input_key` LWE secret key to the `output_key` GLWE secret key.
///
/// # Formal Definition
pub trait LwePackingKeyswitchKeyGenerationEngine<
    InputSecretKey,
    OutputSecretKey,
    LwePackingKeyswitchKey,
>: AbstractEngine where
    InputSecretKey: LweSecretKeyEntity,
    OutputSecretKey: GlweSecretKeyEntity,
    LwePackingKeyswitchKey: LwePackingKeyswitchKeyEntity,
{
    /// Generates a new packing keyswitch key.
    fn generate_new_lwe_packing_keyswitch_key(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> Result<LwePackingKeyswitchKey, LwePackingKeyswitchKeyGenerationError<Self::EngineError>>;

    /// Unsafely generates a new packing keyswitch key.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LwePackingKeyswitchKeyGenerationError`]. For safety concerns _specific_ to an
    /// engine, refer to the implementer safety section.
    unsafe fn generate_new_lwe_packing_keyswitch_key_unchecked(
        &mut self,
        input_key: &InputSecretKey,
        output_key: &OutputSecretKey,
        decomposition_level_count: DecompositionLevelCount,
        decomposition_base_log: DecompositionBaseLog,
        noise: Variance,
    ) -> LwePackingKeyswitchKey;
}
