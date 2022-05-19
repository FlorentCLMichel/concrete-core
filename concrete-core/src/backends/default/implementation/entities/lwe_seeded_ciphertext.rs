use crate::commons::crypto::lwe::LweSeededCiphertext as ImplLweSeededCiphertext;
use crate::commons::math::random::Seed;
use crate::specification::entities::markers::{BinaryKeyDistribution, LweSeededCiphertextKind};
use crate::specification::entities::{AbstractEntity, LweSeededCiphertextEntity};
use concrete_commons::parameters::LweDimension;
#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

/// A structure representing a seeded LWE ciphertext with 32 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweSeededCiphertext32(pub(crate) ImplLweSeededCiphertext<u32>);
impl AbstractEntity for LweSeededCiphertext32 {
    type Kind = LweSeededCiphertextKind;
}
impl LweSeededCiphertextEntity for LweSeededCiphertext32 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn seed(&self) -> Seed {
        self.0.get_seed()
    }

    fn generator_byte_index(&self) -> usize {
        self.0.get_generator_byte_index()
    }
}

/// A structure representing a seeded LWE ciphertext with 64 bits of precision.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LweSeededCiphertext64(pub(crate) ImplLweSeededCiphertext<u64>);
impl AbstractEntity for LweSeededCiphertext64 {
    type Kind = LweSeededCiphertextKind;
}
impl LweSeededCiphertextEntity for LweSeededCiphertext64 {
    type KeyDistribution = BinaryKeyDistribution;

    fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_size().to_lwe_dimension()
    }

    fn seed(&self) -> Seed {
        self.0.get_seed()
    }

    fn generator_byte_index(&self) -> usize {
        self.0.get_generator_byte_index()
    }
}
