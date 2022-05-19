use crate::commons::math::random::Seed;
use crate::specification::entities::markers::{KeyDistributionMarker, LweSeededCiphertextKind};
use crate::specification::entities::AbstractEntity;
use concrete_commons::parameters::LweDimension;

/// A trait implemented by types embodying a seeded LWE ciphertext.
///
/// A seeded LWE ciphertext is a compressed version of a regular LWE ciphertext. It uses a CSPRNG to
/// deterministically generate its mask from a given seed. Because the mask can be regenerated from
/// a seeded CSPRNG, the seeded LWE ciphertext only stores the seed (128 bits) instead of the whole
/// mask which can contain hundreds of u32 or u64. This lightweight seeded LWE ciphertext can be
/// more efficiently sent over the network for example. It can then be decompressed into a regular
/// LWE ciphertext that can be used in homomorphic computations.
///
/// A seeded LWE ciphertext is associated with a
/// [`KeyDistribution`](`LweSeededCiphertextEntity::KeyDistribution`) type, which conveys the
/// distribution of the secret key it was encrypted with.
pub trait LweSeededCiphertextEntity: AbstractEntity<Kind = LweSeededCiphertextKind> {
    /// The distribution of the key the ciphertext was encrypted with.
    type KeyDistribution: KeyDistributionMarker;

    /// Returns the LWE dimension of the ciphertext.
    fn lwe_dimension(&self) -> LweDimension;

    /// Returns the seed used to generate the mask of the LWE ciphertext during encryption.
    fn seed(&self) -> Seed;

    /// Returns the shift used to generate the mask of the LWE ciphertext during encryption.
    fn generator_byte_index(&self) -> usize;
}