# SAM3 Local Change Log

- Upstream source: https://github.com/facebookresearch/sam3
- Local branch inspected: `v0.1.4`
- Summary updated: 2026-05-01

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

### 5. Parameterize Transformer Decoder Resolution and Stride

File:

- `sam3/model_builder.py`

What changed:

- Made `_create_transformer_decoder(...)` accept optional `resolution` and
  `stride` arguments instead of hard-coding `1008` and `14`.
- Threaded the same optional arguments through
  `_create_sam3_transformer(...)`, so callers can override decoder geometry
  from the top-level builder.
- Kept the previous defaults unchanged for existing call sites.

Why:

- SAM3 integration can now adapt decoder geometry without editing the builder
  internals.
- This keeps the local vendored tree closer to the upstream structure while
  still exposing the knobs needed by MDSTL-side model wiring.

Risk / behavior notes:

- Existing users that rely on default SAM3 behavior should see no change,
  because the defaults remain `resolution=1008` and `stride=14`.
- Callers that pass custom values now have a supported path instead of
  patching the source locally.

### 6. Parameterize Builder Position-Cache Resolution

File:

- `sam3/model_builder.py`

What changed:

- Renamed `_create_position_encoding(...)`'s cache-control argument from
  `precompute_resolution` to `resolution`.
- Added `_create_tracker_maskmem_position_encoding(...)` so tracker memory
  encoding can expose its own resolution knob without overloading the visual
  backbone helper's semantics.
- Made `_create_tracker_maskmem_backbone(...)` accept optional
  `resolution=1008` and route it into the new tracker maskmem position-encoding
  helper.
- Made `_create_vision_backbone(...)` accept optional `resolution=1008` and
  route it into `_create_position_encoding(...)`.
- Updated internal builder call sites to the new `resolution=...` keyword.

Why:

- MDSTL's reconstructed SAM3 adapters need to disable eager CUDA position-cache
  precompute during construction and leave first-use cache materialization to
  explicit warmup paths.
- Using `resolution` keeps the naming aligned with the earlier
  `_create_transformer_decoder(...)` patch instead of introducing a separate
  one-off builder keyword.
- Splitting tracker maskmem positional encoding into its own helper keeps the
  visual-backbone and tracker-memory builder layers semantically distinct.

Risk / behavior notes:

- Existing SAM3 builder call sites keep the previous behavior because the new
  defaults remain `resolution=1008` where precompute was previously hard-coded.
- Callers can now pass `resolution=None` to bypass eager position-cache
  precompute without locally rewriting builder internals.

### 7. Add Sparse Prompt Padding-Mask Attention Support

Files:

- `sam3/sam/prompt_encoder.py`
- `sam3/sam/mask_decoder.py`
- `sam3/sam/transformer.py`

What changed:

- Extended `PromptEncoder.forward(...)` with optional
  `enable_dummy_boxes`, defaulting to `True` to preserve upstream SAM3
  behavior.
- `PromptEncoder` remains responsible only for prompt embedding. Sparse-prompt
  padding masks are now assembled outside the prompt encoder so MDSTL can
  manage fixed-slot batching explicitly.
- Extended `MaskDecoder.forward(...)` and `predict_masks(...)` with optional
  `sparse_key_padding_mask`.
- `MaskDecoder` now prepends a valid prefix for SAM output tokens and then
  concatenates the sparse-prompt padding mask before entering the two-way
  transformer.
- Extended `TwoWayTransformer`, `TwoWayAttentionBlock`, `Attention`, and
  `RoPEAttention` with a prompt-token key-padding-mask path separate from the
  existing memory mask path.

Why:

- MDSTL stage-1-enhance needs fixed-slot sparse prompt batching with a prompt
  attention mask, instead of relying on extra `label == -1` tokens as fake
  padding.
- Native SAM3 `label == -1` tokens are semantically meaningful
  `not_a_point_embed` tokens, not ignorable padding. Without an explicit prompt
  key-padding mask, padding to a shared `P_max` changes decoder behavior.
- Keeping native dummy-point insertion as an explicit `PromptEncoder`
  compatibility switch avoids forcing MDSTL to reimplement `boxes is None`
  dummy-point behavior at the integration boundary.

Risk / behavior notes:

- Prompt padding-mask semantics intentionally follow standard PyTorch
  key-padding-mask convention:
  - `True = padding`
  - `False = valid`
- This differs from the earlier local `memory_key_padding_mask` patch, where
  SAM3-compatible memory masks still use:
  - `True = valid`
  - `False = padding`
- The prompt mask is only applied where sparse prompt tokens act as keys/values:
  - sparse-token self-attention
  - image-to-token cross-attention
- The prompt mask is not applied to token-to-image attention, because that path
  attends over image tokens rather than sparse prompt tokens.
- Masked prompt calls are no longer eligible for the FA3 fast path in
  `sam3/sam/transformer.py`; they fall back to SDPA because an additive
  attention mask is now present.
- Existing call sites remain source-compatible because all new mask arguments
  are optional, and old callers that ignore prompt padding continue to get the
  previous behavior.

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
- `py_compile` on modified SAM3 files touched by sparse prompt padding-mask
  attention changes:
  - `sam3/sam/prompt_encoder.py`
  - `sam3/sam/mask_decoder.py`
  - `sam3/sam/transformer.py`
- `py_compile` on unchanged compatibility call sites that exercise the updated
  prompt-encoder / mask-decoder interfaces:
  - `sam3/model/sam1_task_predictor.py`
  - `sam3/model/video_tracking_multiplex.py`
  - `sam3/model/multiplex_mask_decoder.py`
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
