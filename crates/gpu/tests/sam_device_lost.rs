//! The AI mask encoder after its device is lost (as an NVK reset loses it): runs fail with an error instead of a panic
//! or a hang, and nothing is put on that device again, so `lightcraft-segment` encodes on the CPU. Its own executable:
//! the lost device stays lost for the process. `LIGHTCRAFT_SAM_GPU_SOFTWARE=1` runs it on lavapipe.
#![cfg(feature = "sam")]

use lightcraft_gpu::sam::{Norm, VitLayer, VitShape, VitStem};

fn values(len: usize, seed: u32) -> Vec<f32> {
    (0..len).map(|i| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8) as f32 / (1u32 << 24) as f32 * 0.2 - 0.1).collect()
}

fn norm(c: usize) -> Norm {
    Norm { gamma: vec![1.0; c], beta: vec![0.0; c] }
}

#[test]
fn a_lost_device_fails_runs_and_is_not_used_again() {
    let shape = VitShape { grid: 4, patch_values: 12, hidden: 16, heads: 2, mlp: 32, window: 2, eps: 1e-6 };
    let (n, c, d) = (16, 16, 8);
    let stem = || VitStem {
        patch_weight: values(c * 12, 1),
        position: values(n * c, 2),
        norm_pre: norm(c),
        rope_window: (vec![1.0; 4 * d], vec![0.0; 4 * d]),
        rope_global: (vec![1.0; n * d], vec![0.0; n * d]),
    };
    let layer = |i: usize| -> Result<VitLayer, String> {
        Ok(VitLayer {
            norm1: norm(c),
            qkv_weight: values(3 * c * c, 3),
            qkv_bias: values(3 * c, 4),
            out_weight: values(c * c, 5),
            out_bias: values(c, 6),
            norm2: norm(c),
            fc1_weight: values(32 * c, 7),
            fc1_bias: values(32, 8),
            fc2_weight: values(c * 32, 9),
            fc2_bias: values(c, 10),
            global: i == 1,
        })
    };
    let vit = match lightcraft_gpu::sam::vit(shape, &stem(), 2, layer) {
        Ok(v) => v,
        Err(e) => {
            lightcraft_testkit::skip!(Gpu, "no GPU for the AI mask kernels here ({e})");
            return;
        }
    };
    let patches = values(n * 12, 11);
    assert!(vit.run(&patches).is_ok());
    lightcraft_gpu::sam::lose_device();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| vit.run(&patches)));
    assert!(matches!(r, Ok(Err(_))), "a run on a lost device is an error, not a panic");
    assert!(lightcraft_gpu::sam::broken_reason().is_some());
    assert!(lightcraft_gpu::sam::vit(shape, &stem(), 2, layer).is_err(), "nothing new goes on the lost device");
    assert!(lightcraft_gpu::sam::adapter().is_err());
}
