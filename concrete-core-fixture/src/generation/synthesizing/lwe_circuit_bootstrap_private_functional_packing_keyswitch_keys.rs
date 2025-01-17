use crate::generation::prototyping::PrototypesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys;
use crate::generation::{IntegerPrecision, KeyDistributionMarker};
use concrete_core::prelude::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysEntity;

pub trait SynthesizesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys<
    Precision: IntegerPrecision,
    InputKeyDistribution: KeyDistributionMarker,
    OutputKeyDistribution: KeyDistributionMarker,
    LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys,
>:
    PrototypesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys<
    Precision,
    InputKeyDistribution,
    OutputKeyDistribution,
> where
    LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys:
        LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysEntity,
{
    fn synthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
        &mut self,
        prototype: &Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto,
    ) -> LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys;
    fn unsynthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
        &mut self,
        entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys,
    ) -> Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto;
    fn destroy_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
        &mut self,
        entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys,
    );
}

mod backend_default {
    use crate::generation::prototypes::{
        ProtoBinaryBinaryLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32,
        ProtoBinaryBinaryLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
    };
    use crate::generation::synthesizing::SynthesizesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys;
    use crate::generation::{BinaryKeyDistribution, Maker, Precision32, Precision64};
    use concrete_core::prelude::{
        LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32,
        LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
    };

    impl
        SynthesizesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys<
            Precision32,
            BinaryKeyDistribution,
            BinaryKeyDistribution,
            LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32,
        > for Maker
    {
        fn synthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            prototype: &Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto,
        ) -> LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32,
        ) -> Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto {
            ProtoBinaryBinaryLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32(entity)
        }

        fn destroy_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            _entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys32,
        ) {
        }
    }

    impl
        SynthesizesLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys<
            Precision64,
            BinaryKeyDistribution,
            BinaryKeyDistribution,
            LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
        > for Maker
    {
        fn synthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            prototype: &Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto,
        ) -> LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64 {
            prototype.0.to_owned()
        }

        fn unsynthesize_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
        ) -> Self::LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeysProto {
            ProtoBinaryBinaryLweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64(entity)
        }

        fn destroy_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &mut self,
            _entity: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64,
        ) {
        }
    }
}
