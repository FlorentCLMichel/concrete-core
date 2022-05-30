use std::prelude::rust_2021::TryFrom;

use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::fftw::engines::FftwEngine;
use crate::prelude::{
    GlweSecretKey32, GlweSecretKey64,
    LweBootstrapKey32, LweBootstrapKey64, LweSecretKey32, LweSecretKey64,
    LweBootstrapKeyEntity, PolynomialCount, PolynomialSize,
};
use crate::commons::crypto::glwe::GlweList;
use crate::commons::crypto::secret::{GlweSecretKey as ImplGlweSecretKey, GlweSecretKey};
use crate::backends::fftw::private::math::fft::Complex64;
use crate::commons::math::polynomial::PolynomialList;
use crate::commons::math::tensor::{AsRefTensor, IntoTensor};
use crate::prelude::numeric::CastInto;
use crate::specification::engines::{GlweSecretKeyTensorProductEngine,
                                    GlweSecretKeyTensorProductError};

/// # Description:
/// Implementation of [`TensorProductGlweSecretKeyCreationEngine`] for [`FftwEngine`] that operates
/// on 32 bits integers. It outputs a tensor product of the input GLWE secret keys in the standard
/// domain.
impl GlweSecretKeyTensorProductEngine<GlweSecretKey32, GlweSecretKey32, GlweSecretKey32>
for FftwEngine
{    fn glwe_secret_key_tensor_product(
        &mut self,
        input_key1: &GlweSecretKey32,
        input_key2: &GlweSecretKey32,
    ) -> Result<GlweSecretKey32, GlweSecretKeyTensorProductError<Self::EngineError>> {
        GlweSecretKeyTensorProductError::perform_generic_checks(
            input_key1,
            input_key2,
        )?;

    Ok(unsafe { self.create_tensor_product_glwe_secret_key_unchecked(input_key1, input_key2)})


    }

    unsafe fn create_tensor_product_glwe_secret_key_unchecked(
        &mut self,
        input_key1: &GlweSecretKey32,
        input_key2: &GlweSecretKey32,
    ) -> GlweSecretKey32 {
        input_key1.create_tensor_product_key(input_key2)
    }
}



