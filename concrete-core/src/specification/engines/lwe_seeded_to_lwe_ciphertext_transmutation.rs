use super::engine_error;
use crate::prelude::AbstractEngine;

use crate::specification::entities::{LweCiphertextEntity, LweSeededCiphertextEntity};

engine_error! {
    LweSeededToLweCiphertextTransmutationEngineError for LweSeededToLweCiphertextTransmutationEngine @
}

/// A trait for engines transmuting LWE seeded ciphertexts into LWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation moves the existing LWE seeded ciphertext into a
/// LWE ciphertext.
///
/// # Formal Definition
///
/// ## LWE seeded ciphertext to LWE ciphertext transmutation
/// ###### inputs:
/// - $G$: a CSPRNG
/// - $\mathsf{ct} = \left( \mathsf{S} , b\right) \in \mathsf{LWE}^n\_{\vec{s}, G}(
///   \mathsf{pt})\subseteq (\mathbb{N}, \mathbb{Z}\_q)$: a seeded LWE ciphertext
///
/// ###### outputs:
/// - $\mathsf{ct} = \left( \vec{a} , b\right) \in \mathsf{LWE}^n\_{\vec{s}}( \mathsf{pt} )\subseteq
///   \mathbb{Z}\_q^{(n+1)}$: an LWE ciphertext
///
/// ###### algorithm:
/// 1. uniformly sample a vector with the CSPRNG seeded with $\mathsf{S}$, $G\_\mathsf{S}$:
/// $\vec{a}\in\mathbb{Z}^n\_{q, G\_\mathsf{S}}$
/// 2. output $\left( \vec{a} , b\right)$
pub trait LweSeededToLweCiphertextTransmutationEngine<InputCiphertext, OutputCiphertext>:
    AbstractEngine
where
    InputCiphertext: LweSeededCiphertextEntity,
    OutputCiphertext: LweCiphertextEntity,
{
    /// Does the transmutation of the LWE seeded ciphertext key into an LWE ciphertext
    fn transmute_lwe_seeded_ciphertext_to_lwe_ciphertext(
        &mut self,
        lwe_seeded_ciphertext: InputCiphertext,
    ) -> Result<OutputCiphertext, LweSeededToLweCiphertextTransmutationEngineError<Self::EngineError>>;

    /// Unsafely transmutes an LWE seeded ciphertext key into an LWE ciphertext
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweSeededToLweCiphertextTransmutationEngineError`].
    /// For safety concerns _specific_ to an engine, refer to the implementer safety section.
    unsafe fn transmute_lwe_seeded_ciphertext_to_lwe_ciphertext_unchecked(
        &mut self,
        lwe_seeded_ciphertext: InputCiphertext,
    ) -> OutputCiphertext;
}
