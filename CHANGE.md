# SAM3 Local Change Log

- Upstream source: https://github.com/facebookresearch/sam3
- Local branch inspected: `v0.1.2`
- Summary updated: 2026-04-25

This file records the intentional local differences in this vendored SAM3 tree.
It includes committed changes in the nested `thirdparty/sam3` git repository
starting at `4ee5552 enable masked attention`, plus the current uncommitted
MDSTL integration patches.

## Committed Changes Since `enable masked attention`

### `4ee5552 enable masked attention`

Files:

- `sam3/model/decoder.py`
- `sam3/sam/transformer.py`

Purpose:

- Add `memory_key_padding_mask` support to SAM3 RoPE/cross-attention paths.
- Convert `memory_key_padding_mask` from SAM3's convention, `True=valid` and
  `False=padding`, into an additive SDPA attention mask with `-inf` on padded
  memory tokens.
- Thread the mask through `TransformerDecoderLayerv2`,
  `DecoupledTransformerDecoderLayerv2`,
  `TransformerEncoderDecoupledCrossAttention`, `SimpleRoPEAttention`, and SAM
  `Attention` / `RoPEAttention`.
- Disable FA3 use for masked calls, because the existing FA3 call path does not
  accept the additive mask.

Impact:

- Enables fixed-slot / padded-memory execution where invalid memory positions
  must not participate in attention.
- Keeps unmasked calls eligible for the original FA3 path.
- Later local patches remove SAM3-internal SDPA backend policy from these files;
  MDSTL now owns the SDPA policy at the caller boundary.

### `6987e03 update version`

File:

- `sam3/__init__.py`

Purpose:

- Bump local SAM3 package version from `0.1.1` to `0.1.2`.

## Current Uncommitted MDSTL Integration Patches

The current working tree contains additional local patches. They are marked in
source with `# >>> CHANGE: ... <<<`, and in several places the original SAM3
implementation is preserved as comments for direct comparison.

### 1. Keep SDPA Backend Policy Outside SAM3 Internals

Files:

- `sam3/model/decoder.py`
- `sam3/model/model_misc.py`
- `sam3/model/vl_combiner.py`
- `sam3/sam/transformer.py`

What changed:

- Removed direct `torch.backends.cuda.enable_*_sdp(...)` calls from SAM3
  attention implementations.
- Removed SAM3-local `sdpa_kernel(...)` contexts around text encoding and
  decoder attention.
- Left `scaled_dot_product_attention(...)` calls "naked" so the caller can
  choose the backend policy.

Why:

- SAM3 is vendored into MDSTL, which composes multiple model families in one
  Python process. A vendored model should not mutate process-global SDPA flags
  or hide backend choices inside low-level attention modules.
- MDSTL's SAM3 Guider now wraps the bottom-network calls with a mode-aware SDPA
  policy: eager mode allows flash / efficient / math; compile modes use math.

Risk / behavior notes:

- Native SAM3 code that expected the vendored implementation to force-enable
  SDPA backends now depends on the outer caller context or PyTorch defaults.
- MDSTL comparison runs become cleaner because backend choice is visible at the
  integration boundary.

### 2. Scope Dynamo Config Instead of Mutating Process Globals

Files:

- `sam3/perflib/compile.py`
- `sam3/model/sam3_multiplex_tracking.py`
- `sam3/model/sam3_video_inference.py`
- `sam3/model/sam3_tracker_base.py`
- `sam3/model/video_tracking_multiplex.py`
- `sam3/model/vitdet.py`
- `sam3/model/text_encoder_ve.py`
- `sam3/model/maskformer_segmentation.py`

What changed:

- Added scoped helpers in `sam3/perflib/compile.py`:
  - `dynamo_config_context(...)`
  - `wrap_with_dynamo_config(...)`
  - `compile_with_dynamo_config(...)`
  - `SAM3_COMPILE_DYNAMO_CONFIG`
  - `SAM3_TRACKER_COMPILE_DYNAMO_CONFIG`
  - `SAM3_ACT_CKPT_DYNAMO_CONFIG`
- Replaced direct writes such as:
  - `torch._dynamo.config.cache_size_limit = ...`
  - `torch._dynamo.config.accumulated_cache_size_limit = ...`
  - `torch._dynamo.config.capture_scalar_outputs = True`
  - `torch._dynamo.config.suppress_errors = True`
  - `torch._dynamo.config.optimize_ddp = False`
- Wrapped lazy compiled callables so Dynamo config is patched during execution,
  where tracing and recompilation actually happen.

Why:

- `torch.compile(...)` is lazy. Patching only around the constructor is not
  enough; the first trace or a later recompile can happen when the compiled
  callable is invoked.
- Direct Dynamo config writes leak into other MDSTL modules.
- `suppress_errors=True` is especially unsafe for native SAM3-vs-MDSTL
  comparison because it silently hides compile failures behind eager fallback.

Risk / behavior notes:

- SAM3 compile cache sizes and scalar-output capture are preserved locally for
  compiled calls.
- `suppress_errors=True` is intentionally not preserved in the default scoped
  config. Compile failures should be visible during comparison runs.
- Activation-checkpoint `optimize_ddp=False` is kept local to the compiled calls
  that need it.

### 3. Scope TF32 and bf16 Inference Precision

Files:

- `sam3/perflib/compile.py`
- `sam3/model_builder.py`
- `sam3/model/sam3_multiplex_base.py`
- `sam3/model/sam3_multiplex_video_predictor.py`
- `sam3/model/sam3_tracking_predictor.py`

What changed:

- Added scoped backend / inference helpers:
  - `backend_config_context(...)`
  - `sam3_backend_context(...)`
  - `sam3_inference_context(...)`
- Removed import-time TF32 mutation from `model_builder.py`.
- Wrapped the `build_sam3_image_model(...)` returned model's public `forward`
  with scoped SAM3 backend policy. This restores the original TF32 preference
  for direct image-model users without changing process-global backend flags at
  import time.
- Removed module import-time TF32 mutation from `sam3_multiplex_base.py`.
- Removed predictor-init TF32 mutation from
  `sam3_multiplex_video_predictor.py`.
- Replaced long-lived predictor `bf16_context.__enter__()` usage with scoped
  `sam3_inference_context(...)` around public predictor API calls.
- In `Sam3MultiplexTrackerPredictor.__getattr__(...)`, route delegated generator
  methods, such as `propagate_in_video(...)`, through a generator wrapper that
  keeps `sam3_inference_context(...)` open across `yield` points.

Why:

- TF32 and autocast policy are process/thread execution policy, not model
  structure. Setting them at import or construction time makes unrelated models
  in the same process inherit SAM3's choices.
- Native SAM3-vs-MDSTL comparisons should explicitly control and record backend
  precision policy.
- Generator functions do not execute when called; they only create a generator
  object. A scoped wrapper that simply returned `attr(...)` would exit the SAM3
  inference context before video propagation actually ran.

Risk / behavior notes:

- SAM3's original Ampere+ TF32 preference is preserved while scoped SAM3
  predictor APIs run, and while the public forward of a model returned by
  `build_sam3_image_model(...)` runs.
- `build_sam3_image_model(...)` is wrapped with `autocast_dtype=None` on
  purpose. The original image-model builder only enabled TF32 globally; it did
  not add a default bf16 autocast policy for direct image forward calls.
- Delegated generator predictor APIs now enter the scoped SAM3 inference context
  on first iteration and exit when the generator is exhausted or closed, matching
  the actual execution lifetime of video propagation.
- Code that bypasses predictor APIs and calls lower-level model methods directly
  must provide its own precision context, just like MDSTL's SAM3 Guider does.
- `sam3/train/trainer.py` still owns its training-process backend setup. That is
  treated as trainer runtime configuration rather than model import/inference
  side effect.

### 3.1 Default SAM3 Usage Compatibility

The local patches are not "zero behavior change" for every possible standalone
SAM3 use. The intended compatibility boundary is:

- Direct image model usage through `build_sam3_image_model(...)` keeps SAM3's
  original Ampere+ TF32 preference during `model(...)`, but no longer leaves TF32
  enabled for the rest of the Python process.
- Video / multiplex public predictor APIs keep SAM3's original TF32 + bf16
  inference preference while their public request / propagation APIs execute,
  but no longer enter a long-lived autocast context during construction.
- `compile=False` remains the default for the official video/multiplex models,
  so the scoped Dynamo compile changes do not affect the default non-compile
  path.
- `compile=True` is intentionally stricter than the original code because
  `suppress_errors=True` is no longer enabled by default. Compile failures are
  expected to surface instead of silently falling back to eager execution.
- Lower-level module calls that bypass public builders/predictors may observe
  the removal of SAM3's old import-time global TF32/SDPA side effects. Such
  callers should set their own precision/backend context explicitly.

### 4. Compile-Friendlier Tensor / Shape Handling

Files:

- `sam3/model/geometry_encoders.py`
- `sam3/model/model_misc.py`

What changed:

- Replaced a pinned-memory temporary tensor in geometry encoding:
  - original created `torch.tensor(...).pin_memory().to(...)`
  - local patch uses `boxes_xyxy.new_tensor(...)`
- Removed Python shape equality checks for 2D/3D `attn_mask` in
  `multi_head_attention_forward(...)`, leaving only the rank check.

Why:

- The pinned-memory path is hostile to fake tensor / compile tracing.
- The removed `attn_mask.shape != expected_shape` checks can force problematic
  Python shape guards in compiled paths. MDSTL's SAM3 Guider owns static valid
  mask shapes at its boundary.

Risk / behavior notes:

- Invalid attention mask sizes may now fail later inside PyTorch attention
  instead of at SAM3's explicit pre-check.
- This is acceptable for MDSTL's fixed-shape compile paths, but callers outside
  MDSTL should keep input validation at their own boundary if needed.

## MDSTL-Side Contract

MDSTL mirrors these SAM3 changes at the integration boundary:

- `mdstl/models/sam/sam3_guid.py` wraps SAM3 Guider image, prompt, and text
  bottom-network calls with:
  - scoped SAM3 TF32 backend policy
  - mode-aware SDPA backend policy
  - scoped Dynamo compile config
- MDSTL keeps bf16 autocast under its own `dtype_cfg` / `autocast_context(...)`
  policy instead of using SAM3's predictor-level inference context.
- `sub-dynamic` and `sub-deep` compile modes intentionally match SAM3's
  component-level compile style, while `dynamic` and `deep` additionally compile
  MDSTL runner boundaries.

## Validation Notes

Checks run while preparing these changes:

- `py_compile` on modified SAM3 files touched by Dynamo/backend changes.
- `rg` checks for active direct writes to:
  - `torch._dynamo.config.cache_size_limit`
  - `torch._dynamo.config.accumulated_cache_size_limit`
  - `torch._dynamo.config.capture_scalar_outputs`
  - `torch._dynamo.config.suppress_errors`
  - `torch._dynamo.config.optimize_ddp`
  - SAM3 inference/import-time `torch.backends.*` assignments
- Small runtime checks that scoped Dynamo/backend contexts restore the previous
  process state after exit.
- Lightweight generator-wrapper check for `Sam3MultiplexTrackerPredictor`-style
  delegation:
  - regular delegated callables enter and exit `sam3_inference_context(...)`
    during the call;
  - delegated generator callables enter on first `next(...)`, stay inside the
    context across yielded values, and exit on exhaustion or early
    `generator.close()`;
  - wrapped generator callables remain detectable by
    `inspect.isgeneratorfunction(...)`.

Expected remaining backend writes:

- `sam3/perflib/compile.py` writes backend flags only inside scoped context
  managers and restores them.
- `sam3/train/trainer.py` still writes backend flags as part of explicit trainer
  process setup.
