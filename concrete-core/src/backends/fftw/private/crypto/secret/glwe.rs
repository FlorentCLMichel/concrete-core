use std::marker::PhantomData;
use crate::backends::fftw::private::crypto::bootstrap::FourierBuffers;
use crate::backends::fftw::private::math::fft::{AlignedVec, Complex64, FourierPolynomial};
use crate::commons::crypto::glwe::GlweCiphertext;
use crate::commons::crypto::secret::GlweSecretKey;
use crate::commons::math::tensor::{AsMutSlice, AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use crate::commons::math::torus::UnsignedTorus;
use crate::prelude::{GlweDimension, KeyKind, PolynomialCount, PolynomialSize, TensorProductKeyKind};

/// A GLWE secret key in the Fourier Domain.
#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FourierGlweSecretKey<Kind, Cont, Scalar>
    where
    Kind: KeyKind, 
{
    tensor: Tensor<Cont>,
    pub poly_size: PolynomialSize,
    pub kind: PhantomData<Kind>,
    _scalar: std::marker::PhantomData<Scalar>,
}

impl<Kind, Scalar> FourierGlweSecretKey<Kind, AlignedVec<Complex64>, Scalar> {
    /// Allocates a new GLWE secret key in the Fourier domain whose coefficients are all `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::prelude::BinaryKeyKind;
    /// let glwe: FourierGlweSecretKey<BinaryKeyKind, _, u32> =
    ///     FourierGlweSecretKey::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweDimension
    /// (7));
    /// assert_eq!(glwe.glwe_dimension(), GlweDimension(7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn allocate(value: Complex64, poly_size: PolynomialSize, glwe_dimension: GlweDimension) -> 
                                                                                              Self
        where
            Scalar: Copy,
    {
        let mut tensor = Tensor::from_container(AlignedVec::new(glwe_dimension.0 * poly_size.0));
        tensor.as_mut_tensor().fill_with_element(value);
        FourierGlweSecretKey {
            tensor,
            poly_size,
            kind: Kind,
            _scalar: Default::default(),
        }
    }
}

impl<Kind, Cont, Scalar: UnsignedTorus> FourierGlweSecretKey<Kind, Cont, Scalar> {
    /// Creates a GLWE secret key in the Fourier domain from an existing container.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::prelude::BinaryKeyKind;
    ///
    /// let glwe_key: FourierGlweSecretKey<BinaryKeyKind, _, u32> = 
    /// FourierGlweSecretKey::from_container(
    ///     vec![Complex64::new(0., 0.); 7 * 10],
    ///     GlweDimension(7),
    ///     PolynomialSize(10),
    /// );
    /// assert_eq!(glwe.glwe_dimension(), GlweDimension(7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn from_container(cont: Cont, glwe_dimension: GlweDimension, poly_size: PolynomialSize) -> 
                                                                                              Self
        where
            Cont: AsRefSlice,
    {
        let tensor = Tensor::from_container(cont);
        ck_dim_div!(tensor.len() => glwe_dimension().0, poly_size.0);
        FourierGlweSecretKey {
            tensor,
            poly_size,
            kind: Kind,
            _scalar: Default::default(),
        }
    }

    /// Returns the dimension of the GLWE secret key
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::prelude::BinaryKeyKind;
    ///
    /// let glwe: FourierGlweSecretKey<BinaryKeyKind, _, u32> =
    ///     FourierGlweSecretKey::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweDimension(7));
    /// assert_eq!(glwe.glwe_dimension(), GlweDimension(7));
    /// ```
    pub fn glwe_dimension(&self) -> GlweDimension {
        GlweDimension(self.as_tensor().len() / self.poly_size.0)
    }

    /// Returns the size of the polynomials used in the secret key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::prelude::BinaryKeyKind;
    ///
    /// let glwe: FourierGlweSecretKey<BinaryKeyKind, _, u32> =
    ///     FourierGlweSecretKey::allocate(Complex64::new(0., 0.), PolynomialSize(10), GlweDimension
    /// (7));
    /// assert_eq!(glwe.polynomial_size(), PolynomialSize(10));
    /// ```
    pub fn polynomial_size(&self) -> PolynomialSize {
        self.poly_size
    }

    /// Fills a Fourier GLWE secret key with the Fourier transform of a GLWE secret key in
    /// coefficient domain.
    ///
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::bootstrap::FourierBuffers;
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::commons::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::commons::crypto::secret::GlweSecretKey;
    /// use concrete_core::commons::math::random::Seed;
    /// use concrete_core::prelude::BinaryKeyKind;
    /// let mut fourier_glwe_key: FourierGlweSecretKey<BinaryKeyKind, _, u32> =
    ///     FourierGlweSecretKey::allocate(Complex64::new(0., 0.), PolynomialSize(128), GlweDimension(7));
    ///
    /// let mut buffers = FourierBuffers::new(fourier_glwe_key.poly_size, fourier_glwe_key.glwe_size);
    ///
    /// let mut secret_generator = SecretRandomGenerator::<SoftwareRandomGenerator>::new(Seed(0));
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(7),
    ///     PolynomialSize(128),
    ///     &mut secret_generator,
    /// );
    ///
    /// fourier_glwe_key.fill_with_forward_fourier(&secret_key, &mut buffers)
    /// ```
    pub fn fill_with_forward_fourier<InputCont>(
        &mut self,
        glwe_key: &GlweSecretKey<Kind, InputCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        GlweSecretKey<Kind, InputCont>: AsRefTensor<Element = Scalar>,
    {
        // We retrieve a buffer for the fft.
        let fft_buffer = &mut buffers.fft_buffers.first_buffer;
        let fft = &mut buffers.fft_buffers.fft;

        // We move every polynomial to the fourier domain.
        let poly_list = glwe_key.as_polynomial_list();
        let iterator = self.polynomial_iter_mut().zip(poly_list.polynomial_iter());
        for (mut fourier_poly, coef_poly) in iterator {
            fft.forward_as_torus(fft_buffer, &coef_poly);
            fourier_poly
                .as_mut_tensor()
                .fill_with_one((fft_buffer).as_tensor(), |a| *a);
        }
    }

    /// Fills a GLWE secret key with the inverse fourier transform of a Fourier GLWE secret key
    /// ```
    /// use concrete_commons::parameters::{GlweDimension, PolynomialSize};
    /// use concrete_core::backends::fftw::private::crypto::bootstrap::FourierBuffers;
    /// use concrete_core::backends::fftw::private::crypto::secret::FourierGlweSecretKey;
    /// use concrete_core::backends::fftw::private::math::fft::Complex64;
    /// use concrete_core::commons::crypto::secret::generators::SecretRandomGenerator;
    /// use concrete_core::commons::crypto::secret::GlweSecretKey;
    /// use concrete_core::commons::math::random::Seed;
    /// use concrete_core::prelude::BinaryKeyKind;
    ///
    /// let mut fourier_glwe: FourierGlweSecretKey<BinaryKeyKind, _, u32> =
    ///     FourierGlweSecretKey::allocate(Complex64::new(0., 0.), PolynomialSize(128), GlweDimension(7));
    ///
    /// let mut buffers = FourierBuffers::new(fourier_glwe.poly_size, fourier_glwe.glwe_size);
    /// let mut buffers_out = FourierBuffers::new(fourier_glwe.poly_size, fourier_glwe.glwe_size);
    ///
    /// let mut secret_generator = SecretRandomGenerator::<SoftwareRandomGenerator>::new(Seed(0));
    /// let secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(7),
    ///     PolynomialSize(128),
    ///     &mut secret_generator,
    /// );
    ///
    /// fourier_glwe.fill_with_forward_fourier(&secret_key, &mut buffers);
    ///
    /// let mut out_secret_key: GlweSecretKey<_, Vec<u32>> = GlweSecretKey::generate_binary(
    ///     GlweDimension(7),
    ///     PolynomialSize(128),
    ///     &mut secret_generator,
    /// );
    ///
    /// fourier_glwe.fill_with_backward_fourier(&mut out_secret_key, &mut buffers_out);
    /// ```
    pub fn fill_with_backward_fourier<InputCont, Scalar_>(
        &mut self,
        glwe_key: &mut GlweSecretKey<Kind, InputCont>,
        buffers: &mut FourierBuffers<Scalar>,
    ) where
        Cont: AsMutSlice<Element = Complex64>,
        GlweSecretKey<Kind, InputCont>: AsMutTensor<Element = Scalar_>,
        Scalar_: UnsignedTorus,
    {
        // We retrieve a buffer for the fft.
        let fft = &mut buffers.fft_buffers.fft;

        let mut poly_list = glwe_key.as_mut_polynomial_list();

        // we move every polynomial to the coefficient domain
        let iterator = poly_list
            .polynomial_iter_mut()
            .zip(self.polynomial_iter_mut());

        for (mut coef_poly, mut fourier_poly) in iterator {
            fft.backward_as_torus(&mut coef_poly, &mut fourier_poly);
        }
    }

    /// Returns an iterator over references to the polynomials contained in the GLWE key.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::PolynomialSize;
    /// use concrete_core::commons::math::polynomial::PolynomialList;
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// assert_eq!(list.polynomial_iter().count(), 4);
    /// ```
    pub fn polynomial_iter(
        &self,
    ) -> impl Iterator<Item = FourierPolynomial<&[<Self as AsRefTensor>::Element]>>
        where
            Self: AsRefTensor,
    {
        self.as_tensor()
            .subtensor_iter(self.poly_size.0)
            .map(FourierPolynomial::from_tensor)
    }

    /// Returns an iterator over mutable references to the polynomials contained in the list.
    ///
    /// # Example
    ///
    /// ```
    /// use concrete_commons::parameters::{MonomialDegree, PolynomialSize};
    /// use concrete_core::commons::math::polynomial::PolynomialList;
    /// let mut list =
    ///     PolynomialList::from_container(vec![1u8, 2, 3, 4, 5, 6, 7, 8], PolynomialSize(2));
    /// for mut polynomial in list.polynomial_iter_mut() {
    ///     polynomial
    ///         .get_mut_monomial(MonomialDegree(0))
    ///         .set_coefficient(10u8);
    ///     assert_eq!(polynomial.polynomial_size(), PolynomialSize(2));
    /// }
    /// for polynomial in list.polynomial_iter() {
    ///     assert_eq!(
    ///         *polynomial.get_monomial(MonomialDegree(0)).get_coefficient(),
    ///         10u8
    ///     );
    /// }
    /// assert_eq!(list.polynomial_iter_mut().count(), 4);
    /// ```
    pub fn polynomial_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = FourierPolynomial<&mut [<Self as AsMutTensor>::Element]>>
        where
            Self: AsMutTensor,
    {
        let chunks_size = self.poly_size.0;
        self.as_mut_tensor()
            .subtensor_iter_mut(chunks_size)
            .map(FourierPolynomial::from_tensor)
    }

    pub fn create_tensor_product_key<OutputCont>(
        &mut self,
    ) -> GlweSecretKey<TensorProductKeyKind, OutputCont>
        where
            Self: AsRefTensor<Element=Scalar>,
    {
        // .0 accesses the inner value, i.e. the underlying key wrapped in the GlweSecretKey32
        let input_list_1 = self.0.as_polynomial_list();
        let input_list_2 = self.0.as_polynomial_list();

        // TODO do the conversions to the Fourier domain and back like the tensor product on 
        // ciphertexts
        
        // TODO check allocation size
        let mut output_list = PolynomialList::allocate(0 as u32,
                                                       PolynomialCount(glwe_secret_key.0
                                                           .polynomial_size().0),
                                                       glwe_secret_key.0.polynomial_size());

        {
            let mut iter_output = output_list.polynomial_iter_mut();

            // fill the output of the iterator up with the correct product/s
            for (i, polynomial1) in input_list_1.polynomial_iter().enumerate() {
                for (j, polynomial2) in input_list_2.polynomial_iter().enumerate() {
                    let mut output_poly1 = iter_output.next().unwrap();
                    // TODO: correct the below, we need s_i, s_is_j, s_i^2 terms in the same order
                    output_poly1.fill_with_karatsuba_mul(&polynomial1, &polynomial2);
                }
            }
        }
        // TODO match against the key kind
        let tensor_key =
            GlweSecretKey::binary_from_container(output_list.as_tensor().as_slice().to_vec(),
                                                 glwe_secret_key.0.polynomial_size());

        GlweSecretKey(tensor_key)
    }
}