## D1: Correctness tolerance, f32 engine vs reference (2026-09-25)

**Choice:**
we combine max absolute difference and relative difference into one check: |ours - ref| ≤ atol + rtol·|ref|, with **atol = 5e-4** and **rtol = 0**. we check the logits at every position, not just the last one, and every check reports one of three buckets: PASS / PASS near-tie / FAIL (with position, token, ours, ref, how far off). a near-tie is when the winner flipped but the reference's top 2 scores were less than 2 × atol = 1e-3 apart, so rounding alone could explain the flip.

this tolerance is only for the f32 engine vs the reference. quantization (M5) and the GPU (M7) will get their own tolerances, measured the same way.

the tolerance is fixed here in M0, before the engine exists. if a test fails later, we do not loosen the tolerance to make it pass. changing it needs a new decision with a reason.

**Alternatives considered:**
- **same text (llama2.c):** generate greedily and check the text matches. i didn't go with it because it only looks at the winner, so a bug where the winner stays the same but the scores are off by 0.3 passes, and a near-tie that flips from rounding fails for no reason.
- **top-5 overlap (vLLM):** even looser than same text. vLLM can live with it because their fp16 GPU kernels can't match tightly, but we are f32 vs f32 so we can be a lot stricter. the big reason against both of these: this engine becomes the judge for M5 and M7, so if it has a hidden bug, that bug gets copied into the GPU code and the tests still pass.
- **1e-4, aka the 2x factor:** i felt that the margin above the naive engine's worst was too close for comfort as it sat only 2x above. we only measured 20 prompts and run.c, not our own rust engine, so i thought a prompt or an engine we haven't tested yet could land above that line and give us false positives (a correct engine failing the test). i will say however that it gives a larger gap when it comes to the smallest planted bug at 2,500x, but the 10x factor (5e-4) held a good balance between both.
- **5e-3, aka the 100x factor:** still 50x below the smallest planted bug, but we only planted 2 bugs and a real bug could be a lot smaller than those, so it gives small bugs too much room to hide.

**Why:**
we decided to go for atol = 5e-4 because it sits 9x above the naive engine's worst noise and 560x below the smallest planted bug. the naive engine's worst noise is 5.5e-5: that's llama2.c's run.c built with -O3 (plain loops, like our first engine will be) vs PyTorch f32, over 20 prompts × 256 positions × 32000 logits. we used the worst case and not the average, because the average was 13x smaller and one bad logit is enough to change the output.

for the rtol, at this point in time its use is not necessary since the size of the error does not change no matter if the logit is 1 or 28.75 (the biggest one we saw). no logit anywhere was off by more than 5.5e-5. rtol would also loosen the check on the biggest logits, which are the tokens the model actually picks. we will measure it again during the quantization milestone and when we get to a larger model, and only add it if the error turns out to grow with the size of the logit.

**Evidence:** `test_ref/tolerance.md` (what we tested, the results, and how to rerun it). scripts are in `scripts/tolerance/`.
