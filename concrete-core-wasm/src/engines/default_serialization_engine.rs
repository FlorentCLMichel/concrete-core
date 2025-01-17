use crate::*;
use concrete_core::prelude as core;
use concrete_core::prelude::AbstractEngine;
use concrete_core::specification::engines::*;
use js_sys::Uint8Array;
use paste::paste;
use std::panic;
use wasm_bindgen::prelude::*;

macro_rules! implserde {
    ($(($type: ident, $name: ident),)*) => {
        paste!{
        #[wasm_bindgen]
        impl DefaultSerializationEngine{
        $(
                pub fn [< serialize_ $name >](&mut self, entity: &$type) -> JsResult<Uint8Array> {
                    let ser = jserr!(self.0.serialize(&entity.0))?;
                    Ok(Uint8Array::from(ser.as_slice()))
                }

                pub fn [< deserialize_ $name >](&mut self, bytes: &Uint8Array) -> JsResult<$type> {
                    let vec = bytes.to_vec();
                    wrap!($type, self.0.deserialize(vec.as_slice()))
                }
        )*
        }
        }
    };
}

#[wasm_bindgen]
pub struct DefaultSerializationEngine(core::DefaultSerializationEngine);

#[wasm_bindgen]
impl DefaultSerializationEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsResult<DefaultSerializationEngine> {
        panic::set_hook(Box::new(console_error_panic_hook::hook));
        wrap!(
            DefaultSerializationEngine,
            core::DefaultSerializationEngine::new(())
        )
    }
}

implserde! {
    (FloatEncoder, float_encoder),
    (FloatEncoderVector, float_encoder_vector),
    (CleartextF64, cleartext_f64),
    (CleartextVectorF64, cleartext_vector_f64),
    (Cleartext64, cleartext_64),
    (CleartextVector64, cleartext_vector_64),
    (Plaintext64, plaintext_64),
    (PlaintextVector64, plaintext_vector_64),
    (LweCiphertext64, lwe_ciphertext_64),
    (LweCiphertextVector64, lwe_ciphertext_vector_64),
    (LweSecretKey64, lwe_secret_key_64),
    (LweKeyswitchKey64, lwe_keyswitch_key_64),
    (LweBootstrapKey64, lwe_bootstrap_key_64),
    (GlweCiphertext64, glwe_ciphertext_64),
    (GlweCiphertextVector64, glwe_ciphertext_vector_64),
    (GlweSecretKey64, glwe_secret_key_64),
    (Cleartext32, cleartext_32),
    (CleartextVector32, cleartext_vector_32),
    (Plaintext32, plaintext_32),
    (PlaintextVector32, plaintext_vector_32),
    (LweCiphertext32, lwe_ciphertext_32),
    (LweCiphertextVector32, lwe_ciphertext_vector_32),
    (LweSecretKey32, lwe_secret_key_32),
    (LweKeyswitchKey32, lwe_keyswitch_key_32),
    (LweBootstrapKey32, lwe_bootstrap_key_32),
    (GlweCiphertext32, glwe_ciphertext_32),
    (GlweCiphertextVector32, glwe_ciphertext_vector_32),
    (GlweSecretKey32, glwe_secret_key_32),
}
