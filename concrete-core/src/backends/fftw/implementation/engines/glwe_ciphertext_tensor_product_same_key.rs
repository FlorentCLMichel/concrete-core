use concrete_commons::parameters::GlweSize;

use crate::backends::fftw::engines::FftwEngine;
use crate::backends::fftw::entities::{
    FftwFourierGlweCiphertext32, FftwFourierGlweCiphertext64,
};
use crate::backends::fftw::private::crypto::glwe::FourierGlweCiphertext as ImplFourierGlweCiphertext;
use crate::backends::fftw::private::math::fft::Complex64;
use crate::prelude::{FftwFourierGlweTensorProductCiphertext32, FftwFourierGlweTensorProductCiphertext64, GlweCiphertext32, GlweCiphertext64, GlweCiphertextTensorProductSameKeyEngine, GlweCiphertextTensorProductSameKeyError, GlweTensorProductCiphertext32, GlweTensorProductCiphertext64, ScalingFactor};
use crate::specification::entities::GlweCiphertextEntity;

/// # Description:
/// Implementation of [`GlweCiphertextTensorProductSameKeyEngine`] for [`FftwEngine`] that operates
/// on 32-bit integer GLWE Ciphertexts.
impl GlweCiphertextTensorProductSameKeyEngine<GlweCiphertext32, GlweCiphertext32, GlweTensorProductCiphertext32>
    for FftwEngine
{
    fn tensor_product_glwe_ciphertext_same_key(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        scale: ScalingFactor,
    ) -> Result<GlweTensorProductCiphertext32, GlweCiphertextTensorProductSameKeyError<Self::EngineError>> {
        GlweCiphertextTensorProductSameKeyError::perform_generic_checks(input1, input2)?;
        // TODO check the scale is lower or equal to MAX U32
        Ok(
            unsafe {
                self.tensor_product_glwe_ciphertext_same_key_unchecked(input1, input2, scale)
            },
        )
    }

    unsafe fn tensor_product_glwe_ciphertext_same_key_unchecked(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        scale: ScalingFactor,
    ) -> GlweTensorProductCiphertext32 {
        let mut buffers = self.get_fourier_u32_buffer(
            input1.polynomial_size(),
            input1.glwe_dimension().to_glwe_size(),
        );
        // convert the first input GLWE ciphertext to the fourier domain
        let mut fourier_1 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );
        fourier_1.fill_with_forward_fourier(&input1.0, &mut buffers);

        // perform the tensor product
        let output = fourier_1.tensor_product_same_key(&input2.0, scale, &mut buffers);

        GlweTensorProductCiphertext32(output)
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductSameKeyEngine`] for [`FftwEngine`] that operates on 64-bit
/// integer GLWE Ciphertexts.
impl GlweCiphertextTensorProductSameKeyEngine<GlweCiphertext64, GlweCiphertext64, GlweTensorProductCiphertext64>
    for FftwEngine
{
    fn tensor_product_glwe_ciphertext_same_key(
        &mut self,
        input1: &GlweCiphertext64,
        input2: &GlweCiphertext64,
        scale: ScalingFactor,
    ) -> Result<GlweTensorProductCiphertext64, GlweCiphertextTensorProductSameKeyError<Self::EngineError>> {
        GlweCiphertextTensorProductSameKeyError::perform_generic_checks(input1, input2)?;
        Ok(
            unsafe {
                self.tensor_product_glwe_ciphertext_same_key_unchecked(input1, input2, scale)
            },
        )
    }

    unsafe fn tensor_product_glwe_ciphertext_same_key_unchecked(
        &mut self,
        input1: &GlweCiphertext64,
        input2: &GlweCiphertext64,
        scale: ScalingFactor,
    ) -> GlweTensorProductCiphertext64 {
        let mut buffers = self.get_fourier_u64_buffer(
            input1.polynomial_size(),
            input1.glwe_dimension().to_glwe_size(),
        );
        // convert the first input GLWE ciphertext to the fourier domain
        let mut fourier_1 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );
        fourier_1.fill_with_forward_fourier(&input1.0, &mut buffers);

        // perform the tensor product
        let output = fourier_1.tensor_product_same_key(&input2.0, scale, &mut buffers);

        GlweTensorProductCiphertext64(output)
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductSameKeyEngine`] for [`FftwEngine`] that operates on 32-bit
/// integer GLWE Ciphertexts in the Fourier domain.
impl
    GlweCiphertextTensorProductSameKeyEngine<
        FftwFourierGlweCiphertext32,
        FftwFourierGlweCiphertext32,
        FftwFourierGlweTensorProductCiphertext32,
    > for FftwEngine
{
    fn tensor_product_glwe_ciphertext_same_key(
        &mut self,
        input1: &FftwFourierGlweCiphertext32,
        input2: &FftwFourierGlweCiphertext32,
        scale: ScalingFactor,
    ) -> Result<FftwFourierGlweTensorProductCiphertext32, 
        GlweCiphertextTensorProductSameKeyError<Self::EngineError>>
    {
        GlweCiphertextTensorProductSameKeyError::perform_generic_checks(input1, input2)?;
        // TODO check that scale is <= MAX U32
        Ok(
            unsafe {
                self.tensor_product_glwe_ciphertext_same_key_unchecked(input1, input2, scale)
            },
        )
    }

    unsafe fn tensor_product_glwe_ciphertext_same_key_unchecked(
        &mut self,
        input1: &FftwFourierGlweCiphertext32,
        input2: &FftwFourierGlweCiphertext32,
        scale: ScalingFactor,
    ) -> FftwFourierGlweTensorProductCiphertext32 {
        // perform the tensor product (in the fourier domain)
        FftwFourierGlweTensorProductCiphertext32(
            input1
                .0
                .tensor_product_same_key_fourier_input(&input2.0, scale),
        )
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductSameKeyEngine`] for [`FftwEngine`] that operates on 64-bit
/// integer GLWE Ciphertexts in the Fourier domain.
impl
    GlweCiphertextTensorProductSameKeyEngine<
        FftwFourierGlweCiphertext64,
        FftwFourierGlweCiphertext64,
        FftwFourierGlweTensorProductCiphertext64,
    > for FftwEngine
{
    fn tensor_product_glwe_ciphertext_same_key(
        &mut self,
        input1: &FftwFourierGlweCiphertext64,
        input2: &FftwFourierGlweCiphertext64,
        scale: ScalingFactor,
    ) -> Result<FftwFourierGlweTensorProductCiphertext64, GlweCiphertextTensorProductSameKeyError<Self::EngineError>>
    {
        GlweCiphertextTensorProductSameKeyError::perform_generic_checks(input1, input2)?;
        Ok(
            unsafe {
                self.tensor_product_glwe_ciphertext_same_key_unchecked(input1, input2, scale)
            },
        )
    }

    unsafe fn tensor_product_glwe_ciphertext_same_key_unchecked(
        &mut self,
        input1: &FftwFourierGlweCiphertext64,
        input2: &FftwFourierGlweCiphertext64,
        scale: ScalingFactor,
    ) -> FftwFourierGlweTensorProductCiphertext64 {
        // perform the tensor product (in the fourier domain)
        FftwFourierGlweTensorProductCiphertext64(
            input1
                .0
                .tensor_product_same_key_fourier_input(&input2.0, scale),
        )
    }
}
