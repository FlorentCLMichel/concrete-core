use std::prelude::rust_2021::TryFrom;

use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::fftw::engines::FftwEngine;
use crate::prelude::{GlweSecretKey32, GlweSecretKey64, LweBootstrapKey32, LweBootstrapKey64, LweSecretKey32, LweSecretKey64, LweBootstrapKeyEntity, PolynomialCount, PolynomialSize, GlweSecretKeyTensorProductSameKeyError, GlweSecretKeyTensorProductSameKeyEngine};
use crate::commons::crypto::glwe::GlweList;
use crate::commons::crypto::secret::{GlweSecretKey as ImplGlweSecretKey, GlweSecretKey};
use crate::backends::fftw::private::math::fft::Complex64;
use crate::commons::math::polynomial::PolynomialList;
use crate::commons::math::tensor::{AsRefTensor, IntoTensor};
use crate::prelude::numeric::CastInto;
use crate::specification::engines::{GlweSecretKeyTensorProductEngine,
                                    GlweSecretKeyTensorProductError};

/// # Description:
/// Implementation of [`GlweSecretKeyTensorProductSameKeyEngine`] for 
/// [`FftwEngine`] that operates
/// on 32 bits integers. It outputs a tensor product of the input GLWE secret keys in the standard
/// domain.
impl GlweSecretKeyTensorProductSameKeyEngine<GlweSecretKey32, GlweSecretKey32>
for FftwEngine
{    
    // TODO write public documentation
    fn create_tensor_product_glwe_secret_key_same_key(
        &mut self,
        input: &GlweSecretKey32,
    ) -> Result<GlweSecretKey32, GlweSecretKeyTensorProductSameKeyError<Self::EngineError>> {

    Ok(unsafe { self.create_tensor_product_glwe_secret_key_same_key_unchecked(input)})
    
    }

    unsafe fn create_tensor_product_glwe_secret_key_same_key_unchecked(
        &mut self,
        input: &GlweSecretKey32,
    ) -> GlweSecretKey32 {
        input.0.create_tensor_product_key()
    }
}


// TODO add u64 implementations



