use concrete_commons::parameters::GlweSize;

use crate::backends::fftw::engines::FftwEngine;
use crate::backends::fftw::entities::{
    FourierGlweCiphertext32, FourierGlweCiphertext64, GlweCiphertext32, GlweCiphertext64,
};
use crate::backends::fftw::private::crypto::bootstrap::FourierBuffers;
use crate::backends::fftw::private::crypto::glwe::{
    FourierGlweCiphertext as ImplFourierGlweCiphertext,
};
use crate::commons::crypto::glwe::{
    GlweCiphertext as ImplGlweCiphertext,
};
use crate::backends::fftw::private::math::fft::Complex64;
use crate::prelude::{GlweCiphertext32, GlweCiphertext64, GlweCiphertextTensorProductEngine, GlweCiphertextTensorProductError, ScalingFactor};
use crate::specification::entities::GlweCiphertextEntity;

/// # Description:
/// Implementation of [`GlweTensorProductEngine`] for [`FftwEngine`] that operates on 32-bit
/// integer Glwe Ciphertexts.
impl GlweCiphertextTensorProductEngine<GlweCiphertext32, GlweCiphertext32, GlweCiphertext32>
    for FftwEngine
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        scale: ScalingFactor,
    ) -> Result<GlweCiphertext32, GlweCiphertextTensorProductError<Self::EngineError>> {
        GlweCiphertextTensorProductError::perform_generic_checks(input1, input2)?;
        // TODO check the scale is lower or equal to MAX U32
        Ok(unsafe { self.tensor_product_glwe_ciphertext_unchecked(input1, input2, scale) })
    }

    unsafe fn tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &GlweCiphertext32,
        input2: &GlweCiphertext32,
        scale: ScalingFactor,
    ) -> GlweCiphertext32 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u32,
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0 * (3 + input1.glwe_dimension().0) * (1 / 2)),
        );

        //let buffers1 = self.get_fourier_u32_buffer(
        //    input1.polynomial_size(),
        //    input1.glwe_dimension().to_glwe_size(),
        //);

        let mut buffers1 = FourierBuffers::new(input1.0.poly_size, input1.0.size());
        let mut buffers2 = FourierBuffers::new(input2.0.poly_size, input2.0.size());
        let mut buffers3 = FourierBuffers::new(input2.0.poly_size, input2.0.size());

        //let buffers2 = self.get_fourier_u32_buffer(
        //    input1.polynomial_size(),
        //    input1.glwe_dimension().to_glwe_size(),
        //);

        // convert the two GLWE ciphertexts of interest to the fourier domain
        let mut fourier_1 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );

        let mut fourier_2 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );

        fourier_1.fill_with_forward_fourier(&input1.0, &mut buffers1);
        fourier_2.fill_with_forward_fourier(&input2.0, &mut buffers2);

        // perform the tensor product (in the fourier domain)
        fourier_1.tensor_product(&fourier_2, scale);

        // convert the result back to the coefficient domain
        fourier_1.fill_with_backward_fourier(&mut ciphertext, &mut buffers3);

        GlweCiphertext32(ciphertext)
        //ciphertext.convert_glwe_ciphertext(fourier_1);
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductEngine`] for [`FftwEngine`] that operates on 64-bit
/// integer Glwe Ciphertexts.
impl GlweCiphertextTensorProductEngine<GlweCiphertext64, GlweCiphertext64, GlweCiphertext64>
    for FftwEngine
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input1: &GlweCiphertext64,
        input2: &GlweCiphertext64,
        scale: ScalingFactor,
    ) -> Result<GlweCiphertext64, GlweCiphertextTensorProductError<Self::EngineError>> {
        GlweCiphertextTensorProductError::perform_generic_checks(input1, input2)?;
        Ok(unsafe { self.tensor_product_glwe_ciphertext_unchecked(input1, input2, scale) })
    }

    unsafe fn tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &GlweCiphertext64,
        input2: &GlweCiphertext64,
        scale: ScalingFactor,
    ) -> GlweCiphertext64 {
        let mut ciphertext = ImplGlweCiphertext::allocate(
            0u64,
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0 * (3 + input1.glwe_dimension().0) * (1 / 2)),
        );

        //let buffers1 = self.get_fourier_u32_buffer(
        //    input1.polynomial_size(),
        //    input1.glwe_dimension().to_glwe_size(),
        //);

        let mut buffers1 = FourierBuffers::new(input1.0.poly_size, input1.0.size());
        let mut buffers2 = FourierBuffers::new(input2.0.poly_size, input2.0.size());
        let mut buffers3 = FourierBuffers::new(input2.0.poly_size, input2.0.size());

        //let buffers2 = self.get_fourier_u32_buffer(
        //    input1.polynomial_size(),
        //    input1.glwe_dimension().to_glwe_size(),
        //);

        // convert the two GLWE ciphertexts of interest to the fourier domain
        let mut fourier_1 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );

        let mut fourier_2 = ImplFourierGlweCiphertext::allocate(
            Complex64::new(0., 0.),
            input1.polynomial_size(),
            GlweSize(input1.glwe_dimension().0),
        );

        fourier_1.fill_with_forward_fourier(&input1.0, &mut buffers1);
        fourier_2.fill_with_forward_fourier(&input2.0, &mut buffers2);

        // perform the tensor product (in the fourier domain)
        fourier_1.tensor_product(&fourier_2, scale);

        // convert the result back to the coefficient domain
        fourier_1.fill_with_backward_fourier(&mut ciphertext, &mut buffers3);

        //ciphertext.convert_glwe_ciphertext(fourier_1);
        GlweCiphertext64(ciphertext)
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductEngine`] for [`FftwEngine`] that operates on 32-bit
/// integer Glwe Ciphertexts in the Fourier domain.
impl
    GlweCiphertextTensorProductEngine<
        FourierGlweCiphertext32,
        FourierGlweCiphertext32,
        FourierGlweCiphertext32,
    > for FftwEngine
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input1: &FourierGlweCiphertext32,
        input2: &FourierGlweCiphertext32,
        scale: ScalingFactor,
    ) -> Result<FourierGlweCiphertext32, GlweCiphertextTensorProductError<Self::EngineError>> {
        GlweCiphertextTensorProductError::perform_generic_checks(input1, input2)?;
        // TODO check that scale is <= MAX U32
        Ok(unsafe { self.tensor_product_glwe_ciphertext_unchecked(input1, input2, scale) })
    }

    unsafe fn tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &FourierGlweCiphertext32,
        input2: &FourierGlweCiphertext32,
        scale: ScalingFactor,
    ) -> FourierGlweCiphertext32 {
        // perform the tensor product (in the fourier domain)
        FourierGlweCiphertext32(input1.0.tensor_product(&input2.0, scale))
    }
}

/// # Description:
/// Implementation of [`GlweTensorProductEngine`] for [`FftwEngine`] that operates on 64-bit
/// integer Glwe Ciphertexts in the Fourier domain.
impl
    GlweCiphertextTensorProductEngine<
        FourierGlweCiphertext64,
        FourierGlweCiphertext64,
        FourierGlweCiphertext64,
    > for FftwEngine
{
    fn tensor_product_glwe_ciphertext(
        &mut self,
        input1: &FourierGlweCiphertext64,
        input2: &FourierGlweCiphertext64,
        scale: ScalingFactor,
    ) -> Result<FourierGlweCiphertext64, GlweCiphertextTensorProductError<Self::EngineError>> {
        GlweCiphertextTensorProductError::perform_generic_checks(input1, input2)?;
        Ok(unsafe { self.tensor_product_glwe_ciphertext_unchecked(input1, input2, scale) })
    }

    unsafe fn tensor_product_glwe_ciphertext_unchecked(
        &mut self,
        input1: &FourierGlweCiphertext64,
        input2: &FourierGlweCiphertext64,
        scale: ScalingFactor,
    ) -> FourierGlweCiphertext64 {
        // perform the tensor product (in the fourier domain)
        FourierGlweCiphertext64(input1.0.tensor_product(&input2.0, scale))
    }
}
