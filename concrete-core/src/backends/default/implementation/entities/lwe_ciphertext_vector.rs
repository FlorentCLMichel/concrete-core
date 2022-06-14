use crate::commons::crypto::lwe::LweList as ImplLweList;
use crate::specification::entities::markers::{BinaryKeyDistribution, LweCiphertextVectorKind};
use crate::specification::entities::{AbstractEntity, LweCiphertextVectorEntity};
use concrete_commons::parameters::{LweCiphertextCount, LweDimension};
#[cfg(feature = "backend_default_serialization")]
use serde::{Deserialize, Serialize};

/// A structure representing a vector of LWE ciphertexts with 32 bits of precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertextVector32(pub(crate) ImplLweList<Vec<u32>>);

impl AbstractEntity for LweCiphertextVector32 {
    type Kind = LweCiphertextVectorKind;
}

impl LweCiphertextVectorEntity for LweCiphertextVector32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn lwe_ciphertext_count(&self) -> LweCiphertextCount {
        LweCiphertextCount(self.0.count().0)
    }
}

#[cfg(feature = "backend_default_serialization")]
#[derive(Serialize, Deserialize)]
pub(crate) enum LweCiphertextVector32Version {
    V0,
    #[serde(other)]
    Unsupported,
}

/// A structure representing a vector of LWE ciphertexts with 64 bits of precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweCiphertextVector64(pub(crate) ImplLweList<Vec<u64>>);

impl AbstractEntity for LweCiphertextVector64 {
    type Kind = LweCiphertextVectorKind;
}

impl LweCiphertextVectorEntity for LweCiphertextVector64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn lwe_ciphertext_count(&self) -> LweCiphertextCount {
        LweCiphertextCount(self.0.count().0)
    }
}

#[cfg(feature = "backend_default_serialization")]
#[derive(Serialize, Deserialize)]
pub(crate) enum LweCiphertextVector64Version {
    V0,
    #[serde(other)]
    Unsupported,
}
