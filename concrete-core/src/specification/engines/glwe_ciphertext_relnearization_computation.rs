use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::{GlweCiphertextEntity, GlweRelinearizationKeyEntity};

engine_error! {
    GlweCiphertextRelinearizationError for GlweCiphertextRelinearizationEngine@
}

impl<EngineError: std::error::Error> GlweCiphertextRelinearizationError<EngineError> {
    pub fn perform_generic_checks<InputKey, InputCiphertext, OutputCiphertext>(
        input_key: &InputKey,
        input: &InputCiphertext,
        output: &OutputCiphertext,
    ) -> Result<(), Self>
        where
        // TODO: decide on the entity for the Relinearization Key (GLev cts)
            InputKey: GlweRelinearizationKeyEntity,
        // TODO: add trait bounds (using distribution/s on GlweRelinearizationKeyEntity)?
            InputCiphertext: GlweCiphertextEntity,
            OutputCipherext: GlweCiphertextEntity,
    {
        // TODO: once we have the entities we need to check that e.g. the poly sizes in the GLev
        // ciphertext/s which make up the RLK are correct, etc.
        Ok(())

    }
}
/// A trait for engines performing a relinearization on a GLWE ciphertext.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) generates a GLWE ciphertext with
/// the relinearization of the `input` GLWE ciphertexts, using the `input` relinearization key
///
/// # Formal Definition
pub trait GlweCiphertextRelinearizationEngine<InputKey, InputCiphertext, OutputCiphertext>:
AbstractEngine
    where
        InputKey: GlweRelinearizationKeyEntity,
    // TODO: The input ciphertext is the tensor product of two GLWE ciphertexts
        InputCiphertext: GlweCiphertextEntity,
        OutputCiphertext: GlweCiphertextEntity,
{
    fn relinearize_glwe_ciphertext(
        &mut self,
        input_key: &InputKey,
        input_ciphertext: &InputCiphertext,
    ) -> Result<OutputCiphertext, GlweCiphertextRelinearizationError<Self::EngineError>>;

    /// Unsafely performs a relinearization of a GLWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`GlweCiphertextRelinearizationError`]. For safety concerns _specific_ to an engine,
    /// refer to the implementer safety section.

    unsafe fn relinearize_glwe_ciphertext_unchecked(
        &mut self,
        input_key: &InputKey,
        input1: &InputCiphertext,
    )-> OutputCiphertext;
}