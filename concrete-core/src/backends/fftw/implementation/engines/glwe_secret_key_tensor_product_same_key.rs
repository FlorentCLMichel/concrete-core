use crate::backends::fftw::engines::FftwEngine;
use crate::backends::fftw::private::math::fft::Complex64;
use crate::prelude::{GlweDimension, GlweSecretKey32, GlweSecretKeyEntity, GlweSecretKeyTensorProductSameKeyEngine, GlweSecretKeyTensorProductSameKeyError, GlweTensorProductSecretKey32};
use crate::backends::fftw::private::crypto::secret::FourierGlweSecretKey as 
ImplFourierGlweSecretKey;

/// # Description:
/// Implementation of [`GlweSecretKeyTensorProductSameKeyEngine`] for
/// [`FftwEngine`] that operates
/// on 32 bits integers. It outputs a tensor product of the input GLWE secret keys in the standard
/// domain.
impl GlweSecretKeyTensorProductSameKeyEngine<GlweSecretKey32, GlweTensorProductSecretKey32> for FftwEngine {
    // TODO write public documentation (for both 32 and 64)
    fn create_tensor_product_glwe_secret_key_same_key(
        &mut self,
        input: &GlweSecretKey32,
    ) -> Result<GlweTensorProductSecretKey32, GlweSecretKeyTensorProductSameKeyError<Self::EngineError>> {
        Ok(unsafe { self.create_tensor_product_glwe_secret_key_same_key_unchecked(input) })
    }

    unsafe fn create_tensor_product_glwe_secret_key_same_key_unchecked(
        &mut self,
        input: &GlweSecretKey32,
    ) -> GlweTensorProductSecretKey32 {
        let mut buffers = self.get_fourier_u32_buffer(
            input.polynomial_size(),
            input.glwe_dimension().to_glwe_size(),
        );
        // convert the first input GLWE ciphertext to the fourier domain
        let mut fourier_input = ImplFourierGlweSecretKey::allocate(
            Complex64::new(0., 0.),
            input.polynomial_size(),
            GlweDimension(input.glwe_dimension().0),
        );
        fourier_input.fill_with_forward_fourier(&input.0, &mut buffers);

        GlweTensorProductSecretKey32(fourier_input.create_tensor_product_key())
    }
}

/// # Description:
/// Implementation of [`GlweSecretKeyTensorProductSameKeyEngine`] for
/// [`FftwEngine`] that operates
/// on 64 bits integers. It outputs a tensor product of the input GLWE secret keys in the standard
/// domain.
impl GlweSecretKeyTensorProductSameKeyEngine<GlweSecretKey64, GlweSecretKey64> for FftwEngine {
    // TODO write public documentation
    fn create_tensor_product_glwe_secret_key_same_key(
        &mut self,
        input: &GlweSecretKey64,
    ) -> Result<GlweSecretKey64, GlweSecretKeyTensorProductSameKeyError<Self::EngineError>> {
        Ok(unsafe { self.create_tensor_product_glwe_secret_key_same_key_unchecked(input) })
    }

    unsafe fn create_tensor_product_glwe_secret_key_same_key_unchecked(
        &mut self,
        input: &GlweSecretKey64,
    ) -> GlweSecretKey64 {
        input.0.create_tensor_product_key()
    }
}

