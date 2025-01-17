use crate::fixture::Fixture;
use crate::generation::prototyping::{
    PrototypesLweCiphertext, PrototypesLweSecretKey, PrototypesLweSeededCiphertext,
    PrototypesPlaintext,
};
use crate::generation::synthesizing::{
    SynthesizesLweSecretKey, SynthesizesLweSeededCiphertext, SynthesizesPlaintext,
};
use crate::generation::{IntegerPrecision, KeyDistributionMarker, Maker};
use crate::raw::generation::RawUnsignedIntegers;
use crate::raw::statistical_test::assert_noise_distribution;
use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::LweDimension;
use concrete_core::prelude::{
    LweSecretKeyEntity, LweSeededCiphertextEncryptionEngine, LweSeededCiphertextEntity,
    PlaintextEntity,
};

/// A fixture for the types implementing the `LweSeededCiphertextEncryptionEngine` trait.
pub struct LweSeededCiphertextEncryptionFixture;

#[derive(Debug)]
pub struct LweSeededCiphertextEncryptionParameters {
    pub noise: Variance,
    pub lwe_dimension: LweDimension,
}

impl<Precision, KeyDistribution, Engine, Plaintext, SecretKey, Ciphertext>
    Fixture<Precision, (KeyDistribution,), Engine, (Plaintext, SecretKey, Ciphertext)>
    for LweSeededCiphertextEncryptionFixture
where
    Precision: IntegerPrecision,
    KeyDistribution: KeyDistributionMarker,
    Engine: LweSeededCiphertextEncryptionEngine<SecretKey, Plaintext, Ciphertext>,
    Plaintext: PlaintextEntity,
    SecretKey: LweSecretKeyEntity,
    Ciphertext: LweSeededCiphertextEntity,
    Maker: SynthesizesPlaintext<Precision, Plaintext>
        + SynthesizesLweSecretKey<Precision, KeyDistribution, SecretKey>
        + SynthesizesLweSeededCiphertext<Precision, KeyDistribution, Ciphertext>,
{
    type Parameters = LweSeededCiphertextEncryptionParameters;
    type RepetitionPrototypes =
        (<Maker as PrototypesLweSecretKey<Precision, KeyDistribution>>::LweSecretKeyProto,);
    type SamplePrototypes = (
        <Maker as PrototypesPlaintext<Precision>>::PlaintextProto,
        Precision::Raw,
    );
    type PreExecutionContext = (Plaintext, SecretKey);
    type PostExecutionContext = (Plaintext, SecretKey, Ciphertext);
    type Criteria = (Variance,);
    type Outcome = (Precision::Raw, Precision::Raw);

    fn generate_parameters_iterator() -> Box<dyn Iterator<Item = Self::Parameters>> {
        Box::new(
            vec![
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(100),
                },
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(300),
                },
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(600),
                },
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(1000),
                },
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(3000),
                },
                LweSeededCiphertextEncryptionParameters {
                    noise: Variance(0.00000001),
                    lwe_dimension: LweDimension(6000),
                },
            ]
            .into_iter(),
        )
    }

    fn generate_random_repetition_prototypes(
        parameters: &Self::Parameters,
        maker: &mut Maker,
    ) -> Self::RepetitionPrototypes {
        let proto_secret_key = maker.new_lwe_secret_key(parameters.lwe_dimension);
        (proto_secret_key,)
    }

    fn generate_random_sample_prototypes(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::SamplePrototypes {
        let raw_plaintext = Precision::Raw::uniform();
        let proto_plaintext = maker.transform_raw_to_plaintext(&raw_plaintext);
        (proto_plaintext, raw_plaintext)
    }

    fn prepare_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
    ) -> Self::PreExecutionContext {
        let (proto_secret_key,) = repetition_proto;
        let (proto_plaintext, _) = sample_proto;
        let synth_plaintext = maker.synthesize_plaintext(proto_plaintext);
        let synth_secret_key = maker.synthesize_lwe_secret_key(proto_secret_key);
        (synth_plaintext, synth_secret_key)
    }

    fn execute_engine(
        parameters: &Self::Parameters,
        engine: &mut Engine,
        context: Self::PreExecutionContext,
    ) -> Self::PostExecutionContext {
        let (plaintext, secret_key) = context;
        let seeded_ciphertext = unsafe {
            engine.encrypt_lwe_seeded_ciphertext_unchecked(
                &secret_key,
                &plaintext,
                parameters.noise,
            )
        };
        (plaintext, secret_key, seeded_ciphertext)
    }

    fn process_context(
        _parameters: &Self::Parameters,
        maker: &mut Maker,
        repetition_proto: &Self::RepetitionPrototypes,
        sample_proto: &Self::SamplePrototypes,
        context: Self::PostExecutionContext,
    ) -> Self::Outcome {
        let (plaintext, secret_key, seeded_ciphertext) = context;
        let (proto_secret_key,) = repetition_proto;
        let (_, raw_plaintext) = sample_proto;
        let proto_output_seeded_ciphertext =
            maker.unsynthesize_lwe_seeded_ciphertext(seeded_ciphertext);
        let proto_output_ciphertext = maker
            .transform_lwe_seeded_ciphertext_to_lwe_ciphertext(&proto_output_seeded_ciphertext);
        maker.destroy_plaintext(plaintext);
        maker.destroy_lwe_secret_key(secret_key);
        let proto_plaintext =
            maker.decrypt_lwe_ciphertext_to_plaintext(proto_secret_key, &proto_output_ciphertext);
        (
            *raw_plaintext,
            maker.transform_plaintext_to_raw(&proto_plaintext),
        )
    }

    fn compute_criteria(
        parameters: &Self::Parameters,
        _maker: &mut Maker,
        _repetition_proto: &Self::RepetitionPrototypes,
    ) -> Self::Criteria {
        (parameters.noise,)
    }

    fn verify(criteria: &Self::Criteria, outputs: &[Self::Outcome]) -> bool {
        let (means, actual): (Vec<_>, Vec<_>) = outputs.iter().cloned().unzip();
        assert_noise_distribution(&actual, means.as_slice(), criteria.0)
    }
}
