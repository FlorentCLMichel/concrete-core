use crate::backends::fftw::engines::{FftwEngine, FftwError};
use crate::backends::fftw::entities::{FftwFourierGgswCiphertext32, FftwFourierGgswCiphertext64};
use crate::commons::crypto::glwe::GlweCiphertext;
use crate::prelude::{GgswCiphertextEntity, GlweCiphertext32, GlweCiphertext64};
use crate::specification::engines::{
    GlweCiphertextGgswCiphertextExternalProductEngine,
    GlweCiphertextGgswCiphertextExternalProductError,
};
use crate::specification::entities::GlweCiphertextEntity;

impl From<FftwError> for GlweCiphertextGgswCiphertextExternalProductError<FftwError> {
    fn from(err: FftwError) -> Self {
        Self::Engine(err)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextExternalProductEngine`] for [`FftwEngine`] that
/// operates on 32 bits integers.
impl
    GlweCiphertextGgswCiphertextExternalProductEngine<
        GlweCiphertext32,
        FftwFourierGgswCiphertext32,
        GlweCiphertext32,
    > for FftwEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize, DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// // Here a hard-set encoding is applied (shift by 20 bits)
    /// let input_ggsw = 3_u32 << 20;
    /// let input_glwe = vec![3_u32 << 20; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// // Unix seeder must be given a secret input.
    /// // Here we just give it 0, which is totally unsafe.
    /// const UNSAFE_SECRET: u128 = 0;
    /// let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    /// let mut fftw_engine = FftwEngine::new(())?;
    /// let key: GlweSecretKey32 = default_engine.generate_new_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_ggsw = default_engine.create_plaintext_from(&input_ggsw)?;
    /// let plaintext_glwe = default_engine.create_plaintext_vector_from(&input_glwe)?;
    ///
    /// let ggsw = default_engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext_ggsw, noise, level, base_log)?;
    /// let complex_ggsw: FftwFourierGgswCiphertext32 = fftw_engine.convert_ggsw_ciphertext(&ggsw)?;
    /// let glwe = default_engine.encrypt_glwe_ciphertext(&key, &plaintext_glwe, noise)?;
    ///
    /// // Compute the external product.
    /// let product = fftw_engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw)?;
    /// #
    /// assert_eq!(
    /// #     product.polynomial_size(),
    /// #     glwe.polynomial_size(),
    /// # );
    ///
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FftwFourierGgswCiphertext32,
    ) -> Result<GlweCiphertext32, GlweCiphertextGgswCiphertextExternalProductError<Self::EngineError>>
    {
        FftwError::perform_fftw_checks(glwe_input.polynomial_size())?;
        GlweCiphertextGgswCiphertextExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input,
        )?;
        Ok(unsafe {
            self.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input,
            )
        })
    }

    unsafe fn compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext32,
        ggsw_input: &FftwFourierGgswCiphertext32,
    ) -> GlweCiphertext32 {
        let mut output = GlweCiphertext::allocate(
            0u32,
            glwe_input.polynomial_size(),
            glwe_input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u32_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input.0.external_product(
            &mut output,
            &glwe_input.0,
            &mut buffers.fft_buffers,
            &mut buffers.rounded_buffer,
        );
        GlweCiphertext32(output)
    }
}

/// # Description:
/// Implementation of [`GlweCiphertextGgswCiphertextExternalProductEngine`] for [`FftwEngine`] that
/// operates on 64 bits integers.
impl
    GlweCiphertextGgswCiphertextExternalProductEngine<
        GlweCiphertext64,
        FftwFourierGgswCiphertext64,
        GlweCiphertext64,
    > for FftwEngine
{
    /// # Example:
    /// ```
    /// use concrete_commons::dispersion::Variance;
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize, DecompositionBaseLog, DecompositionLevelCount};
    /// use concrete_core::prelude::*;
    /// # use std::error::Error;
    ///
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    /// let glwe_dimension = GlweDimension(2);
    /// let polynomial_size = PolynomialSize(256);
    /// let level = DecompositionLevelCount(1);
    /// let base_log = DecompositionBaseLog(4);
    /// // Here a hard-set encoding is applied (shift by 50 bits)
    /// let input_ggsw = 3_u64 << 50;
    /// let input_glwe = vec![3_u64 << 50; polynomial_size.0];
    /// let noise = Variance(2_f64.powf(-25.));
    ///
    /// // Unix seeder must be given a secret input.
    /// // Here we just give it 0, which is totally unsafe.
    /// const UNSAFE_SECRET: u128 = 0;
    /// let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    /// let mut fftw_engine = FftwEngine::new(())?;
    /// let key: GlweSecretKey64 = default_engine.generate_new_glwe_secret_key(glwe_dimension, polynomial_size)?;
    /// let plaintext_ggsw = default_engine.create_plaintext_from(&input_ggsw)?;
    /// let plaintext_glwe = default_engine.create_plaintext_vector_from(&input_glwe)?;
    ///
    /// let ggsw = default_engine.encrypt_scalar_ggsw_ciphertext(&key, &plaintext_ggsw, noise, level, base_log)?;
    /// let complex_ggsw: FftwFourierGgswCiphertext64 = fftw_engine.convert_ggsw_ciphertext(&ggsw)?;
    /// let glwe = default_engine.encrypt_glwe_ciphertext(&key, &plaintext_glwe, noise)?;
    ///
    /// // Compute the external product.
    /// let product = fftw_engine.compute_external_product_glwe_ciphertext_ggsw_ciphertext(&glwe, &complex_ggsw)?;
    /// #
    /// assert_eq!(
    /// #     product.polynomial_size(),
    /// #     glwe.polynomial_size(),
    /// # );
    ///
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn compute_external_product_glwe_ciphertext_ggsw_ciphertext(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FftwFourierGgswCiphertext64,
    ) -> Result<GlweCiphertext64, GlweCiphertextGgswCiphertextExternalProductError<Self::EngineError>>
    {
        FftwError::perform_fftw_checks(glwe_input.polynomial_size())?;
        GlweCiphertextGgswCiphertextExternalProductError::perform_generic_checks(
            glwe_input, ggsw_input,
        )?;
        Ok(unsafe {
            self.compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
                glwe_input, ggsw_input,
            )
        })
    }

    unsafe fn compute_external_product_glwe_ciphertext_ggsw_ciphertext_unchecked(
        &mut self,
        glwe_input: &GlweCiphertext64,
        ggsw_input: &FftwFourierGgswCiphertext64,
    ) -> GlweCiphertext64 {
        let mut output = GlweCiphertext::allocate(
            0u64,
            glwe_input.polynomial_size(),
            glwe_input.glwe_dimension().to_glwe_size(),
        );
        let buffers = self.get_fourier_u64_buffer(
            ggsw_input.polynomial_size(),
            ggsw_input.glwe_dimension().to_glwe_size(),
        );
        ggsw_input.0.external_product(
            &mut output,
            &glwe_input.0,
            &mut buffers.fft_buffers,
            &mut buffers.rounded_buffer,
        );
        GlweCiphertext64(output)
    }
}
