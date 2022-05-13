use std::prelude::rust_2021::TryFrom;

use concrete_commons::dispersion::Variance;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount};

use crate::backends::fftw::implementation::engines::FftwEngine;
use crate::backends::default::implementation::entities::{
    GlweSecretKey32, GlweSecretKey64,
    LweBootstrapKey32, LweBootstrapKey64, LweSecretKey32, LweSecretKey64,
};
use crate::commons::crypto::bootstrap::{
    StandardBootstrapKey as ImplStandardBootstrapKey,
};
use crate::backends::fftw::implementation::entities::{
    FourierLweBootstrapKey32, FourierLweBootstrapKey64,
};
use crate::backends::fftw::private::crypto::bootstrap::{
    FourierBootstrapKey as ImplFourierBootstrapKey,
};
use crate::commons::crypto::glwe::GlweList;
use crate::commons::crypto::secret::{GlweSecretKey as ImplGlweSecretKey, GlweSecretKey};
use crate::backends::fftw::private::math::fft::Complex64;
use crate::commons::math::polynomial::PolynomialList;
use crate::commons::math::tensor::{AsRefTensor, IntoTensor};
use crate::prelude::{LweBootstrapKeyEntity, PolynomialCount, PolynomialSize};
use crate::prelude::numeric::CastInto;
use crate::specification::engines::{TensorProductGlweSecretKeyCreationEngine,
                                    TensorProductGlweSecretKeyCreationError};

/// # Description:
/// Implementation of [`TensorProductGlweSecretKeyCreationEngine`] for [`FftwEngine`] that operates
/// on 32 bits integers. It outputs a tensor product of the input GLWE secret keys in the standard
/// domain.
impl TensorProductGlweSecretKeyCreationEngine<GlweSecretKey32, GlweSecretKey32, GlweSecretKey32>
for FftwEngine
{    fn create_tensor_product_glwe_secret_key(
        &mut self,
        input_key1: &GlweSecretKey32,
        input_key2: &GlweSecretKey32,
    ) -> Result<GlweSecretKey32, TensorProductGlweSecretKeyCreationError<Self::EngineError>> {
        TensorProductGlweSecretKeyCreationError::perform_generic_checks(
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

        // .0 accesses the inner value, i.e. the underlying key wrapped in the GlweSecretKey32
        let input_list_1 = input_key1.0.as_polynomial_list();
        let input_list_2 = input_key2.0.as_polynomial_list();

        let mut output_list = PolynomialList::allocate(0 as u32,
                                            PolynomialCount(input_key1.0.polynomial_size().0),
                                            input_key1.0.polynomial_size());

        // create iterators over the two polynomial lists, as well as an output
        {
            let mut iter_output = output_list.polynomial_iter_mut();
            let iterator_1 = input_list_1.polynomial_iter();

            // fill the output of the iterator up with the correct product/s
            for (i, polynomial1) in iterator_1.enumerate() {
                for (j, polynomial2) in input_list_2.polynomial_iter().enumerate() {
                    let mut output_poly1 = iter_output.next().unwrap();
                    // TODO: correct the below, we need s_i, s_is_j, s_i^2 terms in the same order
                    output_poly1.fill_with_karatsuba_mul(&polynomial1, &polynomial2);
                }
            }
        }
        let tensor_key =
            ImplGlweSecretKey::binary_from_container(output_list.as_tensor().as_slice().to_vec(),
                                                     input_key1.0.polynomial_size());


        GlweSecretKey32(tensor_key)
    }
}



