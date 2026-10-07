# 06: Rust codebases at DeepSeek (deepseek-recipe, 3FS Rust parts)

Tutor's private reference. The learner designs and types all repo code. Use this file to explain with real code, give pros and cons, and ask questions. **It does not decide anything for the learner.**

## Header

| Repo | Path | Commit | Commit date |
|---|---|---|---|
| deepseek-recipe | `~/refs/inference/deepseek/deepseek-recipe` | `8cadfede7063c896b944e7bae05daa3549ae97ea` | 2026-09-10 (one squashed public commit, so there is no history to mine) |
| 3FS | `~/refs/inference/deepseek/3FS` | `22fca04564c7cc230fd8b9523b8b92864e1dad47` | 2026-05-07 |
| candle (comparison) | `~/refs/inference/candle` | `66a8cf184a5a519671454066b1b9efd446ec9f5c` | 2026-09-24 |
| llama2.c (comparison) | `~/refs/inference/llama2.c` | `350e04fe35433e6d2941dce5a1f53308f87058eb` | 2024-05-29 |
| vLLM (comparison) | `~/refs/inference/vllm` | `22bbe3f1023a68a7d1f2de566dd4242fdcfd36c3` | 2026-09-25 |

Research date: 2026-09-30. All repos were read-only. I checked every `file:line` with `grep -n` or `sed -n`.

**Tags used below:** `[code]` means read in source (ref given). `[docs]` means taken from a repo doc. `[ran]` means I compiled and ran the repo's own file in a scratch crate outside the repo (`state_machine.rs` plus a copy of `StashedChunks`) and saw the output. `[inference]` means my reading or judgement, to be verified before stating as fact.

### Summary (the 12 things worth knowing)

1. **deepseek-recipe is a textbook small Rust workspace.** It has 7 members: 4 library crates, 1 PyO3 `cdylib`, and 2 example binaries marked `publish = false`. It uses `[workspace.package]` and `[workspace.dependencies]`, `resolver = "3"`, edition 2024, and a pinned toolchain. It has **no** `[profile.*]`, **no** `[workspace.lints]`, and no CI files. Its checks live in `docs/development.md:66-71`. `[code]`
2. **Crates are split by what they hide, plus one "narrow waist" types crate.** `deepseek-recipe-core` holds the shared vocabulary types. `-encoding` hides the prompt template. `deepseek-recipe` hides the protocol schemas and the output parser. `-image` hides OpenCV and HTTP behind feature flags. The protocol crate and the encoding crate do not depend on each other; the application wires them together. `[code]`
3. **Errors follow a mixed style.** The dependency-light crates use hand-written enums with `impl Display + Error` (`ConversionError`, `StreamError`, `CalcResizeError`). The heavier crates use `thiserror` (`EncodingError`, `ImageError` with `#[from]`, `#[source]`, `is_retryable()`). `anyhow` appears only in a binary (3FS `src/bin/bench.rs:6`). In Python, errors become custom exceptions that carry `status_code` and `body` attributes. `[code]`
4. **The streaming parser is the most useful code for M4 and post-v1.** It is a byte-level state machine with one KMP matcher per "branch" (end-of-thinking tag, tool-call tag, stop sequences). It holds back only the bytes that could still become a match, and it outputs `(action, byte_len)` segments rather than borrowed slices. A separate stash maps those lengths back onto the original string chunks. I ran worked examples against the real file. `[ran]`
5. **Streaming detokenization in deepseek-recipe:** buffer the token ids, decode the whole buffer, and emit only when the result does not end in U+FFFD (`decoder.rs:58-70`). This works because DeepSeek's tokenizer is byte-level BPE. For SentencePiece-style decoders (TinyStories, TinyLlama), vLLM's prefix/read-offset method or a raw byte buffer is needed. `[code]` `[inference]`
6. **The tokenizer sits behind two one-method traits.** `TokenizerEncoder` and `TokenizerDecoder` are implemented for `tokenizers::Tokenizer` and for `Arc<T>`. Chat templates are **hand-written Rust per model version**, not Jinja. The prompt is rendered as text including the special-token spellings, then tokenized with `add_special_tokens = false`. `[code]`
7. **3FS `chunk_engine` has an aligned buffer written directly with `std::alloc`.** It is `AlignedBuffer(&'static mut [u8])` at 4096-byte alignment (`aligned.rs:6-39`). It has no null check, it hands out uninitialized memory as `&mut [u8]`, the `'static` lifetime is false, and there is no SAFETY comment. Treat it as a counter-example for M1. The same crate shows good patterns: a `thread_local!` scratch buffer, `Drop`-based release, and `#[repr(C)]` with compile-time layout asserts. `[code]` `[inference]`
8. **SAFETY comments in 3FS are inconsistent.** Only 5 SAFETY or `# Safety` comments cover 42 lines containing `unsafe` (26 in chunk_engine, 16 in usrbio-sys and trash_cleaner). One of them cites line numbers that go stale when the file changes (`engine.rs:305-308`). deepseek-recipe has only 2 `unsafe` blocks, and both have SAFETY comments (`opencv.rs:202,436`). `[code]`
9. **3FS links Rust and C++ with `cxx`.** Rust exports functions (`extern "Rust"`), and CMake compiles the generated `.rs.cc` and links the Rust `staticlib`. The `-sys` crate uses `bindgen` in `build.rs` against a C header. No `cbindgen` anywhere. `[code]`
10. **Candle is the M7 blueprint for crate layout.** `candle-kernels` (the `.cu` files, compiled by `build.rs` to PTX) is **excluded** from workspace members (`candle/Cargo.toml:13-17`). It is pulled in only through the `cuda` feature (`candle-core/Cargo.toml:49`). A dummy CUDA backend keeps `Device::Cuda` compiling without that feature (`candle-core/src/lib.rs:113-117`). `[code]`
11. **Tests are inline in both repos.** Both use `#[cfg(test)]` modules, and neither has a `tests/` directory, insta, or proptest. deepseek-recipe has 16 Rust tests plus 24 pytest tests. Its stream state machine, the trickiest code, has **no Rust unit tests** and is covered only through Python. 3FS has 49 Rust tests, one criterion bench, and `#[ignore]` on a test that needs root. `[code]`
12. **Build profiles:** 3FS defines `[profile.release-cmake]` (`debug = true`, `lto = true`), but nothing in the repo invokes it (grep). candle has `[profile.release-with-debug]`. deepseek-recipe has none. These matter for M6, because `perf` needs symbols and cross-crate inlining affects kernels. `[code]`

---

## 1. Repo maps (Rust parts)

### 1.1 deepseek-recipe

```
deepseek-recipe/                     (virtual workspace root: Cargo.toml has [workspace] only)
├── Cargo.toml                       members, resolver 3, [workspace.package], [workspace.dependencies]
├── rust-toolchain.toml              channel 1.97.1 + rustfmt, clippy, rust-src, rust-analyzer
├── rustfmt.toml                     style_edition = "2024" (only setting)
├── deepseek-recipe-core/     lib    shared types: Conversation, InputMessage, ToolDefinition, ImageSource, ImageTokenSpec
├── deepseek-recipe-encoding/ lib    V4 / V4.1 prompt templates + TokenizerEncoder trait
├── deepseek-recipe/          lib    protocol schemas (Anthropic Messages, OpenAI Chat, Responses), request→Conversation,
│                                    stream/ (StateMachine, StreamProcessor, StreamDecoder), SSE event generators
├── deepseek-recipe-image/    lib    image fetch/limits/retry, OpenCV preprocessing behind features
├── deepseek-recipe-python/   cdylib PyO3 bindings, maturin, .pyi stubs, pytest tests
├── encoding-decoding-demo/   bin    axum web demo (publish = false)
├── server-rs/                lib+bin axum SSE server with mock inference (publish = false)
├── server-py/                       FastAPI example (Python)
├── static/tokenizers/{v4,v41}/tokenizer.json   bundled HF tokenizers (used by Python tests)
└── docs/ streaming.md, tokenizer.md, development.md
```

Dependency graph `[code]` (from each crate's `Cargo.toml`):

```
                 deepseek-recipe-core   (serde, serde_json only)
                 ▲        ▲         ▲
   deepseek-recipe   -encoding   -image(features: opencv-preprocess, reqwest-fetch)
   (tokenizers, jsonschema,  (tokenizers,   (tokio, futures, opt: opencv, image, reqwest)
    async-stream, tokio-stream) thiserror)
                 ▲        ▲         ▲
        deepseek-recipe-python (cdylib), server-rs, encoding-decoding-demo   ← wire everything together
```

Size: 13,331 lines of Rust in total (`wc -l`). The biggest files are the Python conversation bindings (880) and the three protocol `convert.rs` files (504-667). `stream/state_machine.rs` is 577 lines and `stream/processor.rs` is 369.

### 1.2 3FS (Rust only)

```
3FS/Cargo.toml                       workspace: members = trash_cleaner, chunk_engine, hf3fs-usrbio-sys
                                     default-members excludes hf3fs-usrbio-sys (it needs a CMake-built .so first)
src/storage/chunk_engine/  lib + staticlib   (7,666 lines)  storage engine called from C++ via cxx
    build.rs                         cxx_build::bridge("src/cxx.rs") (C++ half compiled by CMake)
    src/cxx.rs                       the #[cxx::bridge] + thin raw-pointer adapters
    src/alloc/                       Allocator (Mutex<ChunkAllocator>), Chunk (Drop releases pos), bitset groups
    src/core/engine.rs               Engine: Arc<MetaStore>, LockMap meta cache, DashMap writing list, workers
    src/file/cluster.rs              pread/pwrite, O_DIRECT vs O_SYNC fds, fallocate/punch-hole
    src/meta/                        RocksDB wrapper, merge operator, key encoding
    src/utils/                       AlignedBuffer, Size newtype, ShardsMap, Worker thread, Error/Result
    src/bin/bench.rs                 throughput bench binary (anyhow)
    benches/bench_allocator.rs       criterion bench
    examples/chunk_viewer.rs         clap CLI to inspect RocksDB
src/lib/rs/hf3fs-usrbio-sys/  lib    bindgen over ../../api/hf3fs_usrbio.h + safe-ish wrappers Iov/Ior/RegisteredFd
src/client/trash_cleaner/     bin    single-file CLI (structopt, nix ioctls, tracing json events)
cmake/AddCrate.cmake                 CMake macro: cargo build, link lib<NAME>.a, add target/cxxbridge include dir
```

---

## 2. Deep dive: workspace layout and crate boundaries

### 2.1 deepseek-recipe manifests `[code]`

- **Virtual manifest.** The root `Cargo.toml` has only `[workspace]` (no `[package]`). Members are listed at `Cargo.toml:2-10`, and `resolver = "3"` sits at `:11`. Resolver 3 is the MSRV-aware resolver, and it is the default for edition 2024 packages. In a virtual manifest you must set it explicitly, because there is no root package whose edition could imply it `[inference: Cargo rule]`.
- **`[workspace.package]`** (`Cargo.toml:13-17`) sets `edition = "2024"`, `readme`, `license`, and `repository`. Members inherit these with `edition.workspace = true` (e.g. `deepseek-recipe/Cargo.toml:4`). The Python crate hard-codes `edition = "2024"` instead (`deepseek-recipe-python/Cargo.toml:4`), a small inconsistency.
- **`[workspace.dependencies]`** (`Cargo.toml:19-54`) pins every version once, including **internal crates with both `version` and `path`** (`:24-27`). That combination lets the crates be published to crates.io and still resolve locally. Members write `serde = { workspace = true }`. Features can be added per member: `deepseek-recipe-python/Cargo.toml:18-21` turns image features on, and `server-rs/Cargo.toml:13-16` bypasses the workspace entry with a raw `path` dep plus features.
- **Feature flags** (`deepseek-recipe-image/Cargo.toml:24-30`): `default = ["opencv-preprocess", "reqwest-fetch"]`, with `opencv-preprocess = ["dep:image", "dep:infer", "dep:opencv"]`. The `dep:` syntax keeps optional dependencies from turning into implicit features. The code gates both the module and its re-export: `#[cfg(feature = "opencv-preprocess")] mod opencv;` plus `pub use` (`deepseek-recipe-image/src/lib.rs:17-30`). The docs explain why: a caller that supplies its own preprocessor builds with `--no-default-features` and needs no OpenCV, Clang, or libclang (`docs/development.md:9-14`, `:82-86`). **This is the same problem M7 will have with nvcc.**
- **Examples are workspace members** with `publish = false` (`encoding-decoding-demo/Cargo.toml:6`, `server-rs/Cargo.toml:6`). They compile under `cargo clippy --workspace --all-targets`, so they cannot rot.
- **No `[profile.*]` and no `[workspace.lints]`.** Lints are enforced only by the documented command `cargo clippy --workspace --all-targets --locked -- -D warnings` (`docs/development.md:68`).
- **`server-rs` is lib + bin** in one package. `src/lib.rs` holds the testable request→SSE logic, and `src/main.rs` is a thin axum shell that imports `server_rs::{...}` (`server-rs/src/main.rs:15-19`). This is the "lib + bin" pattern in miniature.

### 2.2 3FS manifests `[code]`

- The root `Cargo.toml:1-22` holds `members`, a **`default-members`** list that leaves out `hf3fs-usrbio-sys` (`:7-10`), `resolver = "2"`, and `[workspace.package]` (`authors`, `edition = "2021"`, `license`, **`rust-version = "1.85.0" # MSRV`** at `:17`). `default-members` means a plain `cargo build` at the root skips the `-sys` crate, whose `build.rs` needs a `.so` that CMake builds first (`hf3fs-usrbio-sys/README`). That is the same trick as candle's `exclude`, done differently. `[inference]`
- **Custom profile** (`Cargo.toml:19-22`): `[profile.release-cmake] debug = true, inherits = "release", lto = true`. **Unused:** `cmake/AddCrate.cmake:1-7` calls `cargo build --release`, and grep finds no `release-cmake` anywhere else.
- **Inheritance is inconsistent.** `chunk_engine/Cargo.toml:4` hard-codes `edition = "2021"` but inherits `license` and `rust-version`. `hf3fs-usrbio-sys/Cargo.toml` inherits nothing.
- **`crate-type = ["lib", "staticlib"]`** (`chunk_engine/Cargo.toml:9`). `staticlib` produces `libchunk_engine.a` for CMake to link (`AddCrate.cmake:16,29`). `lib` keeps the normal Rust `rlib`, so `benches/`, `examples/`, `src/bin/` and unit tests can `use chunk_engine::*`.
- **Loose version requirements** such as `tracing = "0"` and `libc = "0"` (`trash_cleaner/Cargo.toml:12-21`, `chunk_engine/Cargo.toml:12-30`) accept any 0.x, which can include breaking releases. Reproducibility rests entirely on the committed `Cargo.lock`. Treat this as a pitfall to avoid. `[inference]`
- **`[[bench]] name = "bench_allocator" harness = false`** (`chunk_engine/Cargo.toml:40-42`) is the standard criterion setup.

### 2.3 candle for comparison `[code]`

- `candle/Cargo.toml:1-20` lists members `candle-core`, `candle-nn`, `candle-transformers`, `candle-examples`, `candle-pyo3`, and others. It **excludes** `candle-kernels`, `candle-flash-attn`, `candle-metal-kernels`, and `candle-onnx` (`:13-20`), so `cargo build` at the root never needs nvcc. They are still usable as path dependencies: `candle-kernels = { path = "./candle-kernels", ... }` at `:41`.
- `candle-core/Cargo.toml:15` has `candle-kernels = { workspace = true, optional = true }`, and `:49` has `cuda = ["cudarc", "dep:candle-kernels"]`. **The CUDA code lives in two places: a feature inside core (host code, `cuda_backend` module) and a separate crate for the `.cu` files.**
- `candle-core/src/lib.rs:113-117`: `#[cfg(feature = "cuda")] pub use cuda_backend as cuda;` and `#[cfg(not(feature = "cuda"))] pub use dummy_cuda_backend as cuda;`. `Device` (`device.rs:16-20`) and `Storage` (`storage.rs:12-16`) always have a `Cuda` variant; without the feature the dummy types return errors. The enum stays the same across feature sets, and callers never write `#[cfg]`.
- `candle-kernels/build.rs:1-25` uses `cudaforge::KernelBuilder` to scan `src/*.cu`, compile them to PTX with nvcc flags, and write `OUT_DIR/ptx.rs`. `candle-kernels/src/lib.rs:1-3` pulls it in with `include!(concat!(env!("OUT_DIR"), "/ptx.rs"))`.
- `[profile.release-with-debug] inherits = "release", debug = true` (`candle/Cargo.toml:105-107`). No `[workspace.lints]`.
- `candle/.cargo/config.toml` sets `rustflags = ["-C", "target-cpu=native"]`. CI checks an AVX2 build with `RUSTFLAGS="-C target-feature=avx2"` (`.github/workflows/rust-ci.yml:48-52`). This matters for M6: compile-time `target_feature` vs runtime `is_x86_feature_detected!`.

### 2.4 Summary table: manifest features used

| Feature | deepseek-recipe | 3FS | candle |
|---|---|---|---|
| Virtual root manifest | yes | yes | yes |
| `resolver` | "3" | "2" | "2" |
| edition | 2024 | 2021 | 2021 |
| `rust-version` (MSRV) | no (toolchain pinned 1.97.1 instead) | 1.85.0 | no |
| `[workspace.package]` | edition/readme/license/repo | authors/edition/license/rust-version | version/edition/desc/repo/keywords/license |
| `[workspace.dependencies]` | yes, incl. internal path+version | no | yes, incl. internal path+version |
| `[workspace.lints]` / `[lints]` | no | no | no |
| `default-members` / `exclude` | no | default-members | exclude |
| custom profile | none | `release-cmake` (unused) | `release-with-debug` |
| feature-gated heavy deps | image: opencv/reqwest | no | cuda/metal/mkl/accelerate |
| `crate-type` | `cdylib` (python) | `lib + staticlib` | (candle-pyo3 cdylib) |

### 2.5 What each crate hides (Ousterhout lens; see `design_considerations/ousterhout_research.md` ch. 4-5 notes)

| Crate / module | Public surface | What it hides | Deep or shallow |
|---|---|---|---|
| `deepseek-recipe-core` | Plain data types with `pub` fields (`conversation.rs:11-24`, `messages.rs:5-26`) | Almost nothing; it is the **shared vocabulary** | Shallow **on purpose**: a "narrow waist" type crate. It depends only on serde, so every other crate can use it cheaply `[inference]` |
| `-encoding` | trait `PromptEncoding { encode, render_conversation }` (`lib.rs:37-47`) plus two structs with `new`/`with_tokenizer` | Special-token spellings, DSML tool markup, reasoning-effort text, message merging/normalizing, tool-result reordering (`v4/mod.rs`, 435 lines) | **Deep**: 2 methods over the whole template |
| `deepseek-recipe::stream` | `StreamProcessor::{new, with_tokenizer, process}` (`processor.rs:26,41,55`) | Byte-level state machine, KMP, stashing, UTF-8-safe token buffering, stop-sequence and finish-reason logic (~1,000 lines) | **Deep**, with one leak: `pub mod state_machine` (`stream/mod.rs:12`) makes `StateMachine`, `OutputAction`, and `OutputActionSegment` public, and users import `ParsingOptions` from `deepseek_recipe::stream::state_machine` (`deepseek-recipe-python/src/response.rs:19`). The parser's internal vocabulary is on the API path `[inference]` |
| `deepseek-recipe::protocol` | Private `mod protocol` (`lib.rs:12`), re-exported as `pub use protocol::anthropic; pub use protocol::openai;` (`lib.rs:8-9`); schema modules do `pub use schema::*` | Validation and conversion rules per API | Mixed: the schema types are wide (many `pub` fields, because they mirror JSON), and `convert` is deep |
| `-image` | `ImageResolver`, two traits with default "reject" bodies, feature-gated impls | HTTP, retry, budget accounting, OpenCV | Deep, with native deps isolated behind features |
| 3FS `chunk_engine` (Rust API) | `pub use alloc::*; pub use core::*; ...` (`lib.rs:9-18`); `Engine` has **all `pub` fields** (`engine.rs:19-28`) | Nothing from Rust callers | **Shallow/leaky as a Rust API** `[inference]` |
| 3FS `chunk_engine` (C++ API) | ~20 functions in `extern "Rust"` blocks (`cxx.rs:457-582`) | Allocation, RocksDB, copy-on-write, checksums | **Deep.** For this crate, the FFI bridge is the real interface |

**Teaching hook:** ask the learner which of his M1-M8 modules is the "narrow waist" (probably the tensor/buffer type or the `Config`/weights types), and which ones should be deep (`forward`, the tokenizer, the sampler).

---

## 3. The M0 decision: crate layout (neutral pros/cons)

The decision is Luigi's (step 3 of M0 in the current position). Present these one at a time. First the facts that constrain the choice, then the options, then the triggers. Don't recommend.

### 3.1 Facts that constrain the choice (true regardless of preference)

1. **Integration tests, benches, and examples can only `use` a library target.** A bin-only package (`src/main.rs` alone) can have unit tests inside its modules, but `tests/*.rs`, `benches/*.rs` (criterion), and `examples/*.rs` cannot import its code `[inference: Cargo target rules; well established]`. 3FS relies on this: `benches/bench_allocator.rs:1` does `use chunk_engine::*;`, which works because of `crate-type = ["lib", ...]`.
2. **Profiles live only in the root manifest.** Cargo ignores `[profile.*]` in member manifests and warns `[inference: Cargo docs]`. All three repos put profiles at the root.
3. **Crate boundaries affect inlining.** A non-generic function in crate A can be inlined into crate B only if it is `#[inline]`, tiny (the automatic cross-crate inlining of small leaf functions since ~Rust 1.75), or LTO is on `[inference: verify with the Rust version in use]`. For a hot kernel called from another crate, this is a perf decision, not just an organizing one. 3FS's unused `release-cmake` turns on `lto = true` (`3FS/Cargo.toml:22`).
4. **Crates cannot depend on each other in a cycle.** A workspace therefore forces layering (for example, `cli → engine`, never back). Modules inside one crate can reference each other freely.
5. **The orphan rule works per crate.** To `impl SomeTrait for SomeForeignType`, either the trait or the type must be defined in your crate. deepseek-recipe hits this: to implement its own `TokenizerDecoder` for `tokenizers::Tokenizer` (`decoder.rs:21-26`), the `deepseek-recipe` crate must depend on `tokenizers` unconditionally (`deepseek-recipe/Cargo.toml:20`). A later `cuda` crate that wants to implement a core trait for a cudarc type would face the same constraint.
6. **Feature flags are the other way to isolate heavy dependencies.** You can do it without a workspace, as candle does with features inside `candle-core`, or combine both (feature in core plus a separate kernels crate).
7. **A virtual workspace needs `resolver` set explicitly** (deepseek-recipe `Cargo.toml:11`).
8. **Moving from one package to a workspace later is mechanical** (move `src/` into `crates/engine/`, add a root `[workspace]`) `[inference]`. Moving the other way is also easy. The choice is cheap to reverse early and more expensive once `use` paths are everywhere.

### 3.2 Options, with evidence

**A. One package, bin only** (`src/main.rs` + modules)
- Pros: simplest; one `Cargo.toml`; matches llama2.c's single `run.c`; no `pub` API to design.
- Cons: fact 1 means no `tests/` golden-file tests against the reference and no criterion benches without restructuring. M3's "logits vs reference" and M6's benchmark table both want those. Everything in the CLI can reach everything in the engine, so no boundary is enforced.

**B. One package, lib + bin** (`src/lib.rs` + `src/main.rs`, or `src/bin/*.rs`)
- Evidence: `server-rs` (lib.rs + main.rs); `chunk_engine` (lib + `src/bin/bench.rs` + `examples/` + `benches/`).
- Pros: `tests/`, `benches/`, and `examples/` all work. The binary is a thin shell (CLI parsing, printing) over the library, the same split as `server-rs/src/main.rs` vs `lib.rs`. `pub` vs private becomes meaningful: what the binary may touch is the library's API. Still one `Cargo.toml`, one set of dependencies, one `Cargo.lock`.
- Cons: the binary's dependencies (for example a CLI parser) also compile for the library unless marked optional. There is one feature namespace. Everything shares one compile unit, so incremental rebuilds recompile the whole library.

**C. Workspace from day one** (for example `crates/engine` lib + `crates/cli` bin, later `crates/cuda`, `crates/server`)
- Evidence: deepseek-recipe (7 members), candle (9 members plus 6 excluded), 3FS (3).
- Pros: dependency isolation per crate (the CUDA crate's nvcc/cudarc needs, the server's tokio/axum) keeps `cargo test -p engine` fast and GPU-free. Boundaries are enforced by the compiler. `[workspace.dependencies]` and `[workspace.lints]` give one place for versions and lint policy. Crates build in parallel.
- Cons: more manifests and more `pub` decisions before the design is known (CLAUDE.md: "Don't pre-scaffold"). Cross-crate inlining (fact 3). Early crate boundaries may be drawn in the wrong place (temporal decomposition risk; Ousterhout ch. 5). For roughly 2k lines of M1-M4 code, compile-time gains are negligible `[inference]`.

**D. Hybrid: workspace with one member now**
- The root is a virtual manifest from the start with a single `engine` package (lib + bin). New members get added only when a trigger below fires.
- Pros: no restructuring later, and `[workspace.lints]` and profiles have a home from day one. Cons: one more level of directories, and the `resolver` line is mandatory.

### 3.3 Triggers: when a workspace starts paying off (evidence-based)

| Trigger | Milestone | What the repos did |
|---|---|---|
| A crate needs a toolchain other machines may not have (nvcc, OpenCV, libclang) | M7 | candle: `exclude` + optional path dep + feature (`Cargo.toml:13-17`, `candle-core/Cargo.toml:49`); deepseek: features (`deepseek-recipe-image/Cargo.toml:24-30`); 3FS: `default-members` (`Cargo.toml:7-10`) |
| A second consumer of the engine with a very different dependency set (async server: tokio, axum) | post-v1 | deepseek: `server-rs` and `encoding-decoding-demo` are separate `publish = false` members |
| A binding to another language (PyO3 `cdylib`) | side projects | deepseek: `deepseek-recipe-python` is its own crate with `crate-type = ["cdylib"]` (`deepseek-recipe-python/Cargo.toml:9-12`) |
| Shared types used by two otherwise independent parts | M7 / post-v1 | deepseek: `-core` crate is the narrow waist |
| Compile time hurts | M8? | Not evidenced in these repos |

### 3.4 Questions to ask Luigi (one at a time)

1. "Where will the M3 test that compares our logits to the PyTorch reference live, and what does it need to import?" This leads him to fact 1.
2. "In M7, should `cargo test` still pass on a machine without nvcc? How would you arrange that?" This leads to options C/D, `exclude`, `default-members`, and features.
3. "What should the CLI binary be allowed to touch?" This leads to lib vs bin and the `pub` surface.
4. "If `matmul` lives in another crate from `forward`, what happens to inlining?" This leads to fact 3 and LTO.

### 3.5 Profile and lint settings worth raising at M0 or M6 (options, not recommendations)

- `[profile.release] debug = true` (or a custom `profile.profiling` inheriting release, as candle does in `Cargo.toml:105-107` and 3FS in `Cargo.toml:19-22`). `perf` and flamegraph need symbols for M6. `lto = "fat"` or `"thin"` and `codegen-units = 1` often speed up hot loops at the cost of build time. `panic = "abort"` shrinks the binary and removes unwinding paths. None of the three repos sets `panic`. **Every M6 perf claim must say which profile produced it** (CLAUDE.md: "Measure, don't claim").
- `[lints.clippy] undocumented_unsafe_blocks = "deny"` would enforce CLAUDE.md's `// SAFETY:` rule mechanically. It is a clippy restriction lint that checks for a `// SAFETY:` comment on every `unsafe` block. In a workspace, the same setting goes under `[workspace.lints.clippy]` with `lints.workspace = true` in each member. **None of the three repos does this**, and 3FS's missing SAFETY comments show what happens without it. Edition 2024 already warns on `unsafe_op_in_unsafe_fn` `[inference: lint defaults]`.

---

## 4. Deep dive: error handling

### 4.1 What / where `[code]`

| Crate | Error type | Style | Ref |
|---|---|---|---|
| deepseek-recipe | `ConversionError { BadRequest { detail }, Internal { detail } }` | hand-written `Display` + `impl std::error::Error`, constructor helpers `bad_request(impl Into<String>)` | `request/mod.rs:80-109` |
| deepseek-recipe | `StreamError { MissingTokenizer, Decode { detail } }` | hand-written, `Clone + PartialEq` (so tests can compare) | `stream/mod.rs:16-37` |
| deepseek-recipe | HTTP mapping kept **separate** from the error: `status_code()` → 400/500, `into_response()` → OpenAI-style JSON body with `r#type` | extension `impl ConversionError` in another module | `error_response.rs:5-48` (`r#type` at `:8`) |
| -core | `CalcResizeError::NotConverged { max_iter, last_result }` | hand-written (core has no thiserror dependency) | `multimodal/token_spec.rs:22-44` |
| -encoding | `EncodingError { Encode(String), MissingTokenizer }` | `#[derive(thiserror::Error)]` | `lib.rs:26-34` |
| -encoding | Trait returns `Result<Vec<u32>, String>` | **stringly-typed on purpose**: keeps `tokenizers::Error` out of the trait signature, so any tokenizer can implement it | `tokenizer.rs:13` |
| -image | `ImageError` with 17 variants, `#[error(transparent)] TokenBudget(#[from] CalcResizeError)`, `Fetch { url, #[source] source: Box<dyn Error + Send + Sync> }`, method `is_retryable()` | thiserror plus an error-classification method (retryable only for `Fetch`, and for `FetchStatus` with status ≥ 500) | `error.rs:6-111` (`#[from]` `:75`, `#[source]` `:82`, `is_retryable` `:105`) |
| 3FS chunk_engine | one crate-wide `enum Error` (12 variants, mostly `String` payloads), `type Result<T>`, `Display` delegates to `Debug` | hand-written; **no `impl std::error::Error`** | `utils/result.rs:1-22` |
| 3FS bench bin | `anyhow::{Context, Result}`, `.with_context(...)` | anyhow only at the edge | `src/bin/bench.rs:6,25-29` |
| 3FS trash_cleaner | `nix::Result`, `std::process::abort()` on invariant violations | fail-stop | `main.rs:118,263,272,682,694` |

### 4.2 How errors cross boundaries `[code]`

- **Rust → Python:** `pyo3::create_exception!(_native, ConversionError, PyException, "...")` (`deepseek-recipe-python/src/error.rs:7-12`). `request_error()` builds the exception and **attaches `status_code` and `body` attributes** from `error_response.rs` (`error.rs:18-34`), so a FastAPI server can return the right HTTP response. Other failures map to Python built-ins: `RuntimeError` for tokenizer and stream errors (`tokenizer.rs:32`, `response.rs:430-431`); `TypeError`, `OverflowError`, `ValueError` ("circular reference"), and `RecursionError` (depth over 128) in JSON conversion (`json.rs:9,56-57`). Custom `ImageError` and `CalcResizeError` subclass `PyException` and `PyValueError` (`image.rs:17-29`).
- **Rust → C++ (3FS):** a `Pin<&mut CxxString>` out-parameter holds the message, the call returns a null pointer or 0, and **numeric error codes** are written into the request struct: `req.out_error_code = match e { Error::IoError(_) => 4011, ... }` (`cxx.rs:153-166`). The C++ side checks `if (!error.empty())` (`src/storage/store/ChunkEngine.h:85-89`).
- **Side effect of 3FS's missing `std::error::Error` impl** `[inference]`: `bench.rs:39` calls `Engine::open(...).unwrap()` right next to `?`-with-context for the other errors. anyhow's `?` needs `std::error::Error`, which `chunk_engine::Error` does not implement.

### 4.3 Panics vs Results `[code]`

- deepseek-recipe uses `expect("...")` with the message phrased as the **invariant that holds**, not as the failure. Examples: `.expect("JSON values serialize to memory")` (`json_formatter.rs:77-78`) and `.expect("the choice was inserted")` (`chat_completion/response/schema.rs:275`). `unwrap()` appears only in example `main`s (`server-rs/src/main.rs:91-92`) and after an explicit check (`processor.rs:308-309`).
- Tests use `expect_err("...")` plus `assert!(matches!(error, ImageError::TotalSizeTooLarge { size: 11, max: 10 }), "{error}")` (`limits.rs:208-214`).
- trash_cleaner **aborts** rather than risk deleting the wrong thing (`main.rs:255-273`). This is Ousterhout's "just crash" (technique 4) used deliberately for a safety invariant.
- candle, for contrast, returns `Result` for shape errors too (`candle-core/src/error.rs:21-22,77`, `bail!` macro at `:287`).

### 4.4 Applicability by milestone

- **M1 kernels:** shape mismatch in `matmul(out, x, w, n, d)`. Is it a programmer error (`assert_eq!`/`debug_assert!`, as llama2.c effectively does by trusting the caller) or a `Result`, as candle does? This is a real decision with perf (checks in the hot loop) and API consequences. Present both.
- **M2 loading:** a bad checkpoint file is **user input**, so a `Result` is the natural fit. Options: a hand-written enum (like deepseek core, zero dependencies), `thiserror` (like -encoding and -image), or `anyhow` only in `main` (like 3FS bench).
- **M4 CLI:** `main() -> Result<(), Box<dyn Error>>` (as in deepseek's README example, `README.md:91`) versus anyhow at the edge.
- **M7:** CUDA errors (cudarc returns `Result`s) need a variant or a `#[from]`.
- **Post-v1 server:** keep the HTTP mapping separate from the domain error, as `error_response.rs` does, and add an `is_retryable()`-style classification (`ImageError::is_retryable`). That maps onto role project 5 (requeue on worker failure).

### 4.5 Teaching hook and pitfalls

- Hook: "deepseek-recipe uses `thiserror` in two crates and hand-written errors in two others. Why might core avoid the dependency?" (Answer: core is the narrow waist that everyone depends on, so keep it minimal. `[inference]`)
- Pitfall: stringifying errors (`Encode(String)`, 3FS `IoError(String)`) loses the `source()` chain. deepseek does it deliberately at a trait boundary (`tokenizer.rs:13`). 3FS does it everywhere.
- Pitfall: `Display = Debug` (3FS `result.rs:19-22`) produces messages like `InvalidArg("invalid pos")` (the test at `result.rs:30-33` locks this in).

---

## 5. Deep dive: API design

### 5.1 Options / builder structs `[code]`

`ConversionOptions` (`deepseek-recipe/src/request/options.rs:15-70`):
- `#[non_exhaustive]` on a struct with `pub` fields (`:15-16`). Outside crates can read the fields but cannot build the struct with a literal, so new fields are not breaking changes. Construction goes through `new()`/`Default` plus `with_*`.
- `pub const fn new()` (`:36`) means it can be used in `const`/`static`.
- `#[must_use] pub fn with_default_thinking_mode(mut self, v: bool) -> Self` (`:45-50`) is a builder that consumes and returns `self`. `Copy`, so it is cheap.
- `impl Default` delegates to `new()` (`:66-70`).
- `#[non_exhaustive]` on the enum `WebSearchBehavior` too (`:4-5`), so downstream `match` needs a `_` arm.

`ParsingOptions` (`stream/state_machine.rs:31-59`) is the opposite style: all fields `pub`, `Default` implemented, meant to be built with `..Default::default()`. Two styles in one crate.

`with_tokenizer(mut self, t: impl TokenizerEncoder + 'static) -> Self` stores `Option<Box<dyn TokenizerEncoder>>` (`encoding/src/v4/dsv41.rs:19,31-34`; `stream/processor.rs:41-44`). Optional capabilities are attached after construction instead of being constructor parameters. A missing capability fails **at use time** with a typed error (`EncodingError::MissingTokenizer`, `lib.rs:32-33`; `StreamError::MissingTokenizer`, `processor.rs:106-108`).

### 5.2 Traits `[code]`

- **Protocol plugin traits with associated types:** `ProtocolRequest { type Response: ProtocolResponse; fn convert(self, ConversionOptions) -> Result<ConversationRequest, ConversionError>; fn chunk_generator(...) }` (`request/mod.rs:57-76`). `ProtocolResponse: ... + AppendDelta<<Self::ChunkGenerator as ChunkGenerator>::Chunk>` (`response.rs:9-35`), with default methods `chunk_event_type()` → `None` and `done_message()` → `None`. Chat Completions overrides `done_message` to `"[DONE]"` (`openai/chat_completion/response/schema.rs:242-243`).
- **Async trait methods written as return-position `impl Future + Send`:** `fn generate(&mut self, chunk: OutputChunk) -> impl Future<Output = Vec<Self::Chunk>> + Send;` (`stream/mod.rs:68-74`). Implementors may write `async fn generate` (`chat_completion/response/chunk_generator.rs:86`). Spelling out `+ Send` in the trait lets a multi-threaded tokio runtime move the future between threads.
- **Trait methods with default bodies that reject:** `ImageFetcher::fetch` and `ImagePreprocessor::preprocess` default to `Err(ImageError::Unsupported(...))` (`deepseek-recipe-image/src/lib.rs:58-69,82-93`). A test double can be written as `impl ImagePreprocessor for NoPreprocessor {}` (`resolver.rs:282-284`).
- **Internal trait with a public blanket impl:** `pub(crate) trait EncodingV4 { fn system_token(&self) -> &'static str; fn tool_call_tag_name(&self) ...; }` (`v4/mod.rs:71-90`) and `impl<T: EncodingV4> PromptEncoding for T` (`:206`). V4 and V4.1 differ only in about 6 small "hooks" (`dsv41.rs:43-78`), and all shared template logic is written once. Outsiders cannot implement `EncodingV4`, so the hook set can change freely.
- **Blanket impl for `Arc<T>`:** `impl<T: TokenizerDecoder + Sync> TokenizerDecoder for Arc<T>` (`decoder.rs:28-35`), and the same for the encoder (`tokenizer.rs:24-28`). The Python binding shares one `Arc<Tokenizer>` between encoding and processor without cloning the tokenizer (`deepseek-recipe-python/src/encoding.rs:34`).
- **`AppendDelta<D>`** (`util/append_delta.rs:2-39`): a tiny trait that folds a stream of deltas into a complete response (`impl AppendDelta<Option<T>> for Option<T>`, `Vec`, `String`). The same event stream serves both SSE and non-streaming responses. `[code]`

### 5.3 Enums for formats and states `[code]`

- `InferenceChunk { Ready, Text { content, content_tokens }, Token { token_id }, Finish }` (`stream/inference.rs:23-40`) is the backend→parser contract. It accepts **text or token ids**, so a backend that detokenizes itself and one that does not can both plug in. **For M4 and post-v1:** this is exactly the interface between an inference engine and a server.
- `InferenceFinishReason` (backend: Stop/Length/ContentFilter) vs `FinishReason` (after parsing: adds ToolCalls, StopSequence, EndOfStream) (`inference.rs:43-61`). Two enums, because the parser changes the meaning.
- `Option<Option<String>>` for JSON "absent" vs `null` vs value (`chat_completion/response/schema.rs:73,75`).
- Heavy serde attributes on schema enums: `#[serde(tag = "type", rename_all = "snake_case")]` (18 uses), `#[serde(untagged)]` (10), `#[serde(other)]` (6) (grep count).

### 5.4 Newtypes (3FS) `[code]`

- `#[repr(C)] pub struct Size(pub u64)` with `const fn kibibyte/mebibyte` and constants `Size::KB`/`MB`/`GB` (`utils/size.rs:1-37`), plus a macro that generates `From` impls both ways (`:69-79`). **Pitfall:** `From<Size> for u32` is generated with `val.0 as _` (`:79`), which **silently truncates**. `From` is supposed to be lossless; `TryFrom` would be the honest choice `[inference]`.
- `type Bytes = tinyvec::TinyVec<[u8; 28]>` (`utils/bytes.rs:1`) stores small ids inline (no heap allocation for keys of 28 bytes or less).

### 5.5 Applicability

- **M1:** a newtype for dimensions or shapes (like `Size`), and whether to implement `From` or `TryFrom`.
- **M4:** the sampler and generate options. `ConversionOptions` style (`#[non_exhaustive]` + `with_*`) or `ParsingOptions` style (pub fields + `..Default::default()`). Also: tokenizer attached after construction or required in the constructor?
- **M5:** quant formats as an enum (`Q8_0`, `Q4_0`) or as trait impls (the `EncodingV4` hook trait is a model for "same algorithm, different constants").
- **M7:** device abstraction as an enum (candle `Device`) or as a trait. `pub(crate)` trait plus a blanket public trait lets the internal backend trait change without breaking users.

---

## 6. Deep dive: streaming design (M4 streaming detokenization, post-v1 SSE)

`docs/streaming.md` only shows usage (mock backend, `StreamProcessor::new(generator, parsing_options).process(stream)`, lines 37-66). The design lives in the code. There are three layers:

```
InferenceChunk stream ──► [StreamDecoder] ──► text chunks ──► [StateMachine.feed] ──► (action, byte_len) segments
   (Text | Token)          ids → text when a                   byte-level KMP          │
                           character is complete                branches               ▼
                                                        [StashedChunks] maps lengths back onto the String chunks
                                                                   │  → OutputChunk (Raw | Reasoning | ToolCall... )
                                                                   ▼
                                                        [ChunkGenerator] per protocol → SSE events
```

### 6.1 Layer 1: token ids → text (`stream/decoder.rs`) `[code]`

```rust
// decoder.rs:58-70
pub fn decode(&mut self, token_id: u32) -> Result<Option<(String, usize)>, StreamError> {
    self.pending += 1;
    self.ids.push(token_id);
    let content = self.decoder.decode_ids(&self.ids, false)...;
    if content.is_empty() || content.ends_with(REPLACEMENT_CHARACTER) {
        return Ok(None);                                  // keep buffering
    }
    self.ids.clear();
    Ok(Some((content, std::mem::take(&mut self.pending))))  // text + how many ids it covers
}
```

- **Idea:** decode the buffered ids together. The HF ByteLevel decoder turns an incomplete UTF-8 sequence into U+FFFD, so a trailing U+FFFD means "wait for more ids". The Python test `test_split_character_tokens` checks that "🦀" split across several byte tokens comes out whole and that `completion_tokens == len(token_ids)` (`deepseek-recipe-python/tests/test_bindings.py:163-188`).
- `skip_special_tokens = false` on purpose: `</think>` and the DSML tags must reach the state machine (`decoder.rs:56-57`, `docs/tokenizer.md:111-112`).
- Usage accounting: ids still buffered at the end count toward neither text nor usage (`processor.rs:36-38`, `inference.rs:15-18`).
- **Why it works here** `[code + inference]`: the bundled tokenizer is `model: BPE`, `byte_fallback: False`, with decoder `ByteLevel` (checked by parsing `static/tokenizers/v41/tokenizer.json`). ByteLevel decoding of one token needs no neighbours (a leading space is the byte `Ġ`, not a context rule).
- **Weak spots** `[inference]`: (a) the buffer is decoded again from scratch on every step while a character is incomplete. That is cheap because it is at most about 4 ids for UTF-8, **but** (b) an id that decodes to nothing stays in the buffer forever. `test_undecodable_token_ignored` feeds id 999,999,999 (`test_bindings.py:191-213`); later `Token` chunks would be decoded together with that bad id. (c) `pending` always equals `ids.len()`, so the second counter is redundant. (d) A genuinely invalid byte sequence in the middle of the model output (followed by more text) is emitted with its U+FFFD, which is fine. But a trailing invalid byte at end of stream is dropped silently.

**Comparison: vLLM** (`vllm/tokenizers/detokenizer_utils.py:176-270`) keeps `prefix_offset` and `read_offset` and decodes `tokens[prefix_offset:]` vs `tokens[prefix_offset:read_offset]`, emitting the difference. The context tokens "defeat cleanup algorithms in the decode which decide to add a space or not depending on the surrounding ids" (`:195-196`). It uses the same `endswith("�")` wait rule (`:261-265`). vLLM v1 prefers the `tokenizers` library's native `DecodeStream` (`vllm/v1/engine/detokenizer.py:23,182`).

**Comparison: llama2.c** (`run.c:418-443`): `decode(t, prev_token, token)` strips the leading space after BOS (`:421`) and maps `<0x..>` pieces to raw single bytes (`:425-427`). `safe_printf` then **drops any single-byte piece that is not `isprint`/`isspace`** (`:436-440`). In the C locale, bytes ≥ 0x80 are not printable, so **characters that come out as several byte tokens are silently lost** `[inference from code]`. TinyStories is mostly ASCII, so this rarely shows. That is a concrete improvement Luigi's M4 can make and measure.

**For M4 (llama2.c `tokenizer.bin`, SentencePiece with `<0x00>`..`<0xFF>` byte tokens):** three designs for Luigi to choose from:
1. **Byte buffer:** decode each token to bytes (`<0xNN>` → one byte, others → their UTF-8 bytes), push them into a `Vec<u8>`, and emit the longest valid UTF-8 prefix. Rust's `std::str::from_utf8` error tells you exactly how much is valid (`valid_up_to()`) and whether the tail is merely incomplete (`error_len() == None`). No U+FFFD heuristic is needed, and it works per token without re-decoding.
2. **deepseek style:** buffer ids and re-decode until the result has no trailing U+FFFD. This needs a `decode(&[u32]) -> String` that is lossy.
3. **vLLM style:** prefix and read offsets. Only needed when a token's text depends on its neighbours (SentencePiece leading-space rules, e.g. an HF Llama `tokenizer.json` with a `Strip` decoder at M8) `[inference: verify on the M8 tokenizer]`.

**Toy for the learner** (a parallel example, not engine code; compiled and clippy-clean in scratch, output shown):

```rust
struct Utf8Stream { pending: Vec<u8> }

impl Utf8Stream {
    /// Feed raw bytes; return only the text that is complete so far.
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => { out.push_str(s); self.pending.clear(); return out; }
                Err(e) => {
                    let (good, rest) = self.pending.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(good).unwrap());
                    match e.error_len() {
                        None => { self.pending = rest.to_vec(); return out; } // incomplete: wait
                        Some(n) => { out.push('\u{FFFD}'); self.pending = rest[n..].to_vec(); } // garbage
                    }
                }
            }
        }
    }
}
// "🦀" fed one byte at a time → "", "", "", "🦀";  [0xFF,'x',0xE7] → "�x";  [0x8C,0xAB] → "猫"
```

Borrow-checker teaching point in the toy: `good` and `rest` borrow `self.pending`, yet `self.pending = rest.to_vec()` compiles. The right-hand side makes an owned `Vec` first, the borrow ends, and then the assignment happens. Ask Luigi to predict whether it compiles before showing it.

### 6.2 Layer 2: the state machine (`stream/state_machine.rs`) `[code]`

**Data model:**
- `enum Stage { Common{is_leading}, Json, MatchedJson, ToolCalls, ToolName, ToolCallArguments{is_leading}, ToolCallParamName, ToolCallParamType, ToolCallParamValue{string}, Reasoning{is_leading}, Finished }` (`:134-146`).
- Each stage owns a list of **match branches**. A branch is `(MatchState, next_stage, action_on_matched)` (`:149-153`) and matches either a literal (`MatchPattern::String`), "any number of newlines then a literal" (`NewlinesAndString`), or a single leading newline (`LeadingNewline`) (`:445-449`). Each stage also has one `action_on_unmatched` for ordinary bytes (`:206-211`).
- `State::new(stage, options)` (`:217-385`) is the **transition table written as code**. For example, `Stage::Reasoning` gets a branch "newlines + `</think>`" → `Common{is_leading: true}` with action `Skip` (`:357-374`). `Common` gets stop-sequence branches → `Finished`/`StopSequence`, tool-call begin `<｜DSML｜` → `ToolCalls`/`Skip`, and a stray `</think>` → `SkipInvalid{ExtraEndOfThinking}` (`:227-266`).
- Every literal gets a **KMP failure table** (`kmp_table` `:481-498`, `kmp_next` `:519-527`), so a partial match that fails falls back correctly (for example, pattern "猫猫" on input "猫b猫猫").

**The algorithm** (`State::feed` `:387-430`):
1. For each byte, increment `stashed_size` (bytes not yet assigned an action) and feed the byte to every branch. **The first branch to complete wins** (`break` at `:399`), so branch order is priority order.
2. On a match, emit `action_on_unmatched` for the bytes before the match (`stashed_size - matching_len`) and `action_on_matched` for the match itself. Then **replace the whole state**: `*self = State::new(next_stage, options)` (`:412`).
3. At the end of the chunk, hold back `max(matching_len over branches)` bytes (a possible match prefix) and emit the rest as `action_on_unmatched` (`:415-428`).
4. `finish()` flushes held-back bytes as unmatched (`:432-442`).

**Output is `Vec<OutputActionSegment { action, len }>`** (`:121-125`): lengths in bytes, not `&str`. The parser borrows nothing from its input, so it can keep partial state across chunks without lifetimes, and the caller keeps ownership of the chunk `String`s.

**UTF-8 safety** `[inference, checked by the runs below]`: segment boundaries are either the start of a match or the end of a pattern. Patterns are valid UTF-8 whose first byte is ASCII or a UTF-8 lead byte, and in valid UTF-8 input such a byte always starts a character. Chunks are `&str`, so their ends are character boundaries too. So `String::split_off(len)` in the stash (`processor.rs:214`), **which panics on a non-boundary**, never hits the middle of a character. The self-synchronising property of UTF-8 carries this.

### 6.3 Worked example (run against the real file) `[ran]`

Options: `reasoning_initial_stage = Some(ReasoningStage::Start)` (thinking mode) and `stop_sequences = ["END"]`. The backend streams five text chunks:

| Feed | State machine segments (bytes) | Stash output (what the client sees) | Why |
|---|---|---|---|
| `"\nHmm, a story."` | `Skip 1`, `Reasoning 13` | `Reasoning "Hmm, a story."` | `Reasoning{is_leading}` has a `LeadingNewline` branch, so the first `\n` is dropped. The rest is reasoning. |
| `"\n</th"` | *(none)* | *(none)* | `NewlinesAndString("</think>")` has matched `\n` + `</th`, so all 5 bytes are held back. |
| `"ink>\n\nOnce upon"` | `Skip 9`, `Skip 1`, `Skip 1`, `Raw 9` | `Raw "Once upon"` | `\n</think>` completes (9 bytes across two chunks) → `Common{is_leading: true}`. Its `LeadingNewline` eats `\n` twice, then answer text. |
| `" a time. THE E"` | `Raw 13` | *(none!)* | The stop branch "END" matched `E`, so 1 byte is held. See the stash note below. |
| `"ND. More"` | `StopSequence 3`, `SkipInvalid{ContentAfterFinished} 6` | `Raw " a time. THE "`, then the stop | "END" completes across the chunk boundary → `Finished`. Everything after is discarded. The processor records `stop_sequence = "END"` and breaks (`processor.rs:132-135`); finish reason is `StopSequence` (`:151-152`). |

**Stash behaviour (layer 3, `processor.rs:176-358`)** `[ran with a sync copy of StashedChunks]`: `apply_actions` merges consecutive segments with the same action and pops only (a) chunks the pending action covers **whole**, or (b) a partial chunk **when the action changes**. So in feed 4, `Raw 13` covers only 13 of 14 bytes, and " a time. THE " is **held one extra chunk**. It is emitted in feed 5 when the action changes to `StopSequence`. Visible latency is at most one source chunk (one token at a time from a token stream). Worth pointing out as a latency/fragmentation trade-off. `[inference: intent not documented]`

**Tool-call example** (one chunk, `reasoning_initial_stage = Some(Content)`) `[ran]`:

```
input : Let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>
output: Raw "Let me check."  ToolCallBegin  ToolCall{name:"get_weather", args:""}
        ArgsDelta "{"  "\"city\""  ": "  "\""  "Paris"  "\""  "}"      → arguments = {"city": "Paris"}
```

The parameter name is copied through **with its quotes** (`ToolCallParamName` → `RawToolCallArguments{string:false}`, `:324-331`), so it is already a JSON key. String values are JSON-escaped on the fly (`escape_json_string`, `processor.rs:364-369`). Tool-call arguments therefore **stream as JSON deltas without buffering the whole call**. The markup format is the model's template (`encoding/src/v4/mod.rs:303-327` tells the model to emit it).

**Partial-match release** `[ran]`: `["ok <", "｜Ass", "istant｜> fine"]` gives `Raw 3` (hold `<`), then `Raw 7` (`<｜A` failed to match `<｜DSML｜`, so the held `<` is released together with `｜Ass`), then `Raw 15`. With stop sequence `"猫猫"` on `["a猫", "b猫", "猫c"]`: `Raw 1`, `Raw 4` (released "猫b", held the new "猫"), `StopSequence 6`, `SkipInvalid 1`. All lengths fall on character boundaries.

### 6.4 Push vs pull (Python adapter) `[code]`

Rust `StreamProcessor::process` is **pull-based**: it takes `impl Stream<Item = InferenceChunk>` and returns a `Stream` built with `async_stream::stream!` (`processor.rs:55-66`). Python wants **push** (`processor.push(chunk)` → list of events). The adapter (`deepseek-recipe-python/src/response.rs:269-324,419-462`) works like this:
- a `futures::channel::mpsc::channel(1)` sender for input (`:298`),
- a `poll_fn` wrapper that sets an `AtomicBool input_pending` whenever the input receiver is `Pending` (`:299-307`),
- `drain()` runs `block_on(poll_fn(...))` and polls the event stream until it is `Pending` **because it needs input**, then returns the collected events (`:419-446`),
- a `frozen` pyclass holding a `Mutex<Processor>`, with the work run inside `py.detach(...)` (PyO3's GIL release) (`:282-285,326-338`).

**M4 and post-v1 relevance:** Luigi's generate loop will produce tokens (push: "here is token t"), while a server wants an iterator or stream of text deltas (pull). This adapter is one working bridge between the two. The simpler one for M4 is a synchronous `struct` with `fn push(&mut self, token: u32) -> String` (like the toy) that the loop calls. Present both; he picks.

### 6.5 SSE (post-v1) `[code]`

`server-rs/src/lib.rs:170-199` maps each protocol chunk to `format!("event: {event_type}\ndata: {data}\n\n")`, or `data: ...\n\n` for unnamed events. It appends `data: [DONE]\n\n` for Chat Completions (`done_message`, `openai/chat_completion/response/schema.rs:242-243`), and serves it as `Body::from_stream` with `text/event-stream` (`server-rs/src/main.rs:72-80`). The same `ChunkGenerator` output also accumulates into a complete JSON response through `AppendDelta` (`complete_response`, `lib.rs:213`).

### 6.6 Pitfalls to raise

- **Split a `String` only at a character boundary.** `split_off` and `&s[a..b]` panic otherwise. deepseek's design is safe for the reason above. A token-level design that slices by token byte offsets may not be.
- **Stop sequences can span chunks** ("E" + "ND"). The partial match must be held back, not emitted, or the client sees text that should have been cut.
- **Which text do stop sequences apply to?** In deepseek, not to reasoning or tool markup (`state_machine.rs:43-46`). For M4: stop on the EOS/BOS token id (llama2.c stops when `next == 1`, `run.c:763`) vs stop strings. Two different mechanisms.
- **Counting tokens vs text:** `content_tokens` travels with the text (`inference.rs:30-33`). TTFT and decode tok/s (the M4 checkpoint) count **tokens**, not text deltas. A text delta may be empty while a character is incomplete.
- **No Rust unit tests for `state_machine.rs`/`processor.rs`** (grep: tests only in core, image, and the Python tests). The trickiest code is tested only end to end. `[code]`

---

## 7. Deep dive: tokenizer integration (`docs/tokenizer.md`, `encoding/src/tokenizer.rs`)

### 7.1 What `[code]` `[docs]`

- Crate: HF **`tokenizers` 0.23.2** with feature `http` (`Cargo.toml:51`) for `from_pretrained` (`deepseek-recipe-python/src/tokenizer.rs:47-61`). candle uses `tokenizers` 0.23.1 with `default-features = false` (`candle/Cargo.toml:95`).
- **Nothing loads a tokenizer implicitly.** The caller attaches one (`docs/tokenizer.md:5-9`). Without one, `encode` returns `MissingTokenizer`. Bundled copies live in `static/tokenizers/{v4,v41}/tokenizer.json`, and a Hub tokenizer must spell the special tokens identically and map the image token to the same id (`docs/tokenizer.md:11-16`).
- **Two-step encode:** `render_conversation` → prompt `String` (including `<｜begin▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<think>`/`</think>`, and DSML text) and `image_sources`. Then `encode_ids(prompt)` calls `tokenizer.encode(text, false)`, i.e. **`add_special_tokens = false`**, "because the prompt already contains the special token text" (`docs/tokenizer.md:55-58`, `tokenizer.rs:16-21`, `v4/mod.rs:206-213`).
- **Chat template = Rust code, per model version.** Special-token constants are in `v4/mod.rs:14-34`. The template is `render_message` (`:118-204`), which puts a system token at index 0, a user token (or `\n\n` when merging consecutive user/tool turns), assistant + `<think>`...`</think>` + content + tool calls + EOS, and appends the generation prompt `<｜Assistant｜>` + `<think>`/`</think>` at the end (`:245-266`). `normalize_messages` merges or rewrites messages that the template cannot express (`:332-382`). Version differences are hooks in `EncodingV4` (`dsv41.rs:43-78`).
- A **token id constant** leaks into core: `IMAGE_SPECIAL_TOKEN_ID: u32 = 129264` (`multimodal/mod.rs:19`). It is tied to one tokenizer file, and the docs warn about it (`docs/tokenizer.md:14-16`).
- Checked in `tokenizer.json` `[ran: python json parse]`: `<｜begin▁of▁sentence｜>` id 0 and `<｜end▁of▁sentence｜>` id 1 are `special: true`. `<｜User｜>` 128803, `<think>` 128821, and `</think>` 128822 are added tokens with `special: false`, so they **survive `skip_special_tokens = true`** as well. Base vocab 128,000, 1,283 added tokens.
- **Injection surface** `[inference]`: only the image placeholder spelling is rejected in user text (`protocol/validation.rs:8-18`). Other special-token spellings (e.g. a user typing `<｜Assistant｜>`) go through `render → encode(text)` and would be matched as added tokens. This is a known class of chat-template prompt injection. Verify before stating it as a vulnerability.

### 7.2 Implications for Luigi's milestones

- **M4 (TinyStories `tokenizer.bin`, llama2.c format):** no HF crate; Luigi writes encode and decode. The deepseek design still transfers: **put the tokenizer behind a small trait** (`encode_ids`/`decode_ids`, `tokenizer.rs:11-14`, `decoder.rs:16-19`) so that M8 can plug in an HF tokenizer without touching `generate`. Is that worth designing at M4 or only at M8? His call ("Don't pre-scaffold").
- **BOS handling:** llama2.c's `encode(..., bos=1, eos=0, ...)` adds the BOS id directly. deepseek puts BOS **in the text** and disables auto-special-tokens. In M8, mixing these gives double BOS. Make one layer own BOS.
- **Streaming decode:** see §6.1. TinyStories uses SentencePiece byte fallback (`<0x..>`), so option 1 (byte buffer) or option 3 (context offsets) applies. The deepseek option 2 works only because ByteLevel decoding has no neighbour dependence.
- **M8 chat template:** one choice is hand-written Rust per model (deepseek), another is evaluating the Jinja `chat_template` from `tokenizer_config.json` (e.g. with the `minijinja` crate, not used in these repos `[inference]`). Test the rendered prompt against the model's reference output. deepseek tests `encode(conversation) == tokenizer.encode(render(conversation))` (`test_bindings.py:54-64`). A golden prompt-string test would be stronger.
- **TTFT link:** prompt rendering and tokenization come **before** prefill and are part of TTFT. Measure them separately (role project 4).

---

## 8. Deep dive: performance-relevant Rust in 3FS `chunk_engine`

### 8.1 Memory allocation and aligned buffers `[code]`

```rust
// utils/aligned.rs:4-25  (quoted)
pub const ALIGN_SIZE: Size = Size::new(4096);
pub struct AlignedBuffer(&'static mut [u8]);
impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        Self(unsafe {
            let size = std::cmp::max(size, 1).next_multiple_of(ALIGN_SIZE.into());
            let layout = Layout::from_size_align_unchecked(size, ALIGN_SIZE.into());
            let ptr = std::alloc::alloc(layout);
            std::slice::from_raw_parts_mut(ptr, size)
        })
    }
}
impl Drop for AlignedBuffer { fn drop(&mut self) { unsafe { /* same layout */ std::alloc::dealloc(...) } } }
impl std::ops::Deref for AlignedBuffer { type Target = [u8]; ... }   // :28-35
```

What is right: the size is rounded up to a multiple of the alignment, so O_DIRECT I/O works on whole blocks. `Drop` uses the **same layout** it allocated with (a requirement of `dealloc`). `Deref<Target = [u8]>` lets callers use every slice method. Helpers `is_aligned_buf`/`is_aligned_io` (`:47-57`) choose the I/O path.

What to **critique** (a good M1 exercise: "find 4 problems") `[inference, grounded in std docs]`:
1. **No null check.** `std::alloc::alloc` returns null on failure. The standard fix is `if ptr.is_null() { std::alloc::handle_alloc_error(layout) }`.
2. **Uninitialized memory exposed as `&mut [u8]`.** Safe code can read it, e.g. `create_aligned_buf(n)[0]`. Reading uninitialized bytes is undefined behaviour. Fixes are `alloc_zeroed`, a `MaybeUninit<u8>` slice, or an API that only hands out the buffer after it has been written. The crate fills with zeros when it needs to (`chunk.rs:17-21`) or overwrites by `pread` (`chunk.rs:77-83`), but the **type** does not enforce either.
3. **`&'static mut` is a false lifetime.** The memory is freed in `Drop`, so it is not `'static`. It works only because `Deref` reborrows with a shorter lifetime. A raw `NonNull<u8>` + `len` is the honest field type.
4. **No `// SAFETY:` comments** on either `unsafe` block, which violates Luigi's own CLAUDE.md rule.

Also: `thread_local!` gives each thread a **64 MiB** scratch buffer (`chunk.rs:25-27`, `CHUNK_SIZE_ULTRA` = 64 MiB in `types/constants.rs`), allocated lazily on first use. It avoids allocating on the hot path, at a memory cost of threads × 64 MiB.

**Map to M1 (tensor/buffer type, AVX2 later):**
- AVX2 wants 32-byte alignment only for **aligned** loads (`_mm256_load_ps`), which fault on misaligned addresses. Unaligned loads (`_mm256_loadu_ps`) accept any address. `Vec<f32>` guarantees only 4-byte alignment.
- The options to present: (a) `Vec<f32>` + unaligned SIMD loads (simplest; measure the difference in M6); (b) a custom aligned allocation like `AlignedBuffer` but sound (the 4 fixes above); (c) `Vec<A>` where `#[repr(align(32))] struct A([f32; 8])` (alignment by type, no `unsafe` allocation, but lengths must be multiples of 8); (d) a crate (such as `aligned-vec`, not in these repos `[inference]`).
- **M2 interaction** `[code]`: llama2.c's weights start at byte **28** of the file (`sizeof(Config)` = 7 × 4, `run.c:19-27,160`). An mmap'd page-aligned file therefore gives weights that are 4-byte aligned but **never 32-byte aligned**. Aligned loads straight from an mmap would fault. That pushes toward unaligned loads or copying into an aligned buffer. candle handles a similar case by checking alignment before casting `&[u8] → &[T]` and copying otherwise (`candle-core/src/safetensors.rs:115-135`, with SAFETY comments at `:119,128`).
- **M7:** device buffers are a different allocator (cudaMalloc) behind the same idea: RAII `Drop` frees them, and the host side needs pinned (page-locked) memory for fast async copies. That is the GPU counterpart of 3FS USRBIO's shared-memory iov (`hf3fs_usrbio.h:14-27`) `[inference]`.

### 8.2 `unsafe` inventory and SAFETY comments `[code]`

Grep over chunk_engine, usrbio-sys, and trash_cleaner finds 42 lines containing `unsafe` (blocks, `unsafe fn`, `unsafe impl`, bridge declarations) and **5 `SAFETY`/`# Safety` comments** (`cxx.rs:381-383`, `engine.rs:173-177`, `engine.rs:305-308`, `engine.rs:801-804`, `bin/bench.rs:87`). Examples to quote:

- Good: a documented `unsafe fn` contract:
  ```rust
  // engine.rs:171-182
  /// Intentionally leaks Arc pointers to avoid shutdown overhead.
  /// # Safety
  /// This function **must only be called** when the process is guaranteed to exit immediately afterwards. ...
  pub unsafe fn speed_up_quit(&self) { let _ = Arc::into_raw(self.meta_cache.clone()); ... }
  ```
  **Debatable** `[inference]`: leaking memory is *safe* in Rust (`std::mem::forget` is safe), so `unsafe` is used here as a "danger" label rather than for a memory-safety contract. Worth a quiz question.
- Good but **fragile**: `// SAFETY: req.data pointer must be valid ... (lines 386-394, 398-404, 418-424) ...` (`engine.rs:305-308`). **Line numbers in comments go stale** as the file changes.
- A raw pointer smuggled as an integer: `data: u64` in the cxx struct, with the SAFETY comment on the field (`cxx.rs:381-384`), turned back into a slice with `std::slice::from_raw_parts(req.data as *const _, req.length as usize)` (`engine.rs:309-310`).
- **Missing SAFETY:** `transmute(&self.bits)` from `&[u64; 4]` to `&[u8; 32]` (`types/group_state.rs:114-120`). This is sound for alignment (u8 is less strict than u64), **but the bytes are persisted to RocksDB**, so the on-disk format depends on the machine's endianness `[inference]`. `u64::to_le_bytes` or `bytemuck` would make that explicit. **Relevant to M2**: the llama2.c checkpoint is little-endian f32, and reading it with a pointer cast assumes a little-endian host.
- **Missing SAFETY + lifetime extension:** `std::mem::transmute::<&[u8], &[u8]>(chunk.meta().etag.as_slice())` in a test, used to satisfy a `&'static [u8]` field (`engine.rs:1308-1310`; the `&'static [u8]` fields are at `cxx.rs:388-389`).
- **Layout-dependent transmutes guarded by compile-time asserts:** `unsafe { std::mem::transmute(self.meta()) }` from `&ChunkMeta` to `&ffi::RawMeta` (`cxx.rs:54`). It relies on `#[repr(C)]` on `ChunkMeta` (`types/chunk_meta.rs:6-19`) with `RawMeta`'s fields as a prefix, plus `static_assertions::const_assert_eq!(align_of::<ChunkMeta>(), align_of::<ffi::RawMeta>())` (`cxx.rs:585-604`). Note that only **align** is asserted for this pair; the prefix-field compatibility is by convention. On Rust 1.96 a crate is not needed: `const _: () = assert!(size_of::<A>() == size_of::<B>());` `[inference]`.
- deepseek-recipe's only two `unsafe` blocks both have SAFETY comments: `// SAFETY: the allocation is owned by the returned matrix and is fully written by the loop below.` (`deepseek-recipe-image/src/opencv.rs:202-204`, `:436-438`).

### 8.3 Concurrency primitives `[code]`

| Primitive | Where | Use |
|---|---|---|
| `Mutex<ChunkAllocator>` inside `Arc<Allocator>` | `alloc/allocator.rs:4-8,24-38` | One lock around the allocator state; `self: &Arc<Self>` receiver so a `Chunk` can hold an `Arc` back to its allocator |
| `Drop for Chunk` → `allocator.dereference(pos)` | `alloc/chunk.rs:302-306` | RAII: dropping the last `Arc<Chunk>` frees the disk position (README "Use `Arc` to manage ownership of chunk position", `README.md:17-19`) |
| `lockmap::LockMap` (per-key locks, 256 shards, 1<<20 capacity) | `core/engine.rs:22,45` | Metadata cache keyed by chunk id |
| `DashMap<Bytes, HashMap<...>>` | `alloc/writing_chunk.rs:10` | Concurrent writing list |
| Hand-rolled `ShardsMap<K, V, const S: usize = 64>` | `utils/shards_map.rs:8-80` | `[HashMap; S]` picked by hash; `[(); S].map(\|_\| Default::default())` builds the array (`:23`) |
| `AtomicU64` metrics swapped to 0 on read | `alloc/metrics.rs:3-10`, `cxx.rs:267-308` | Lock-free counters; uses `AcqRel` where `Relaxed` would do for independent counters `[inference]` |
| `AtomicBool` + `Condvar` worker threads | `utils/worker.rs:17-101` | Background allocate/compact loops |
| `std::thread::spawn` stress tests | `meta/rocksdb.rs:237-260`, `core/engine.rs:1689-1696` | 16 threads × 1000 ops, then verify |
| `thread_local!` `RefCell<AlignedBuffer>` | `alloc/chunk.rs:25-27,77-83` | Per-thread scratch |

No rayon, no crossbeam, no tokio in chunk_engine (grep). deepseek-recipe uses tokio (async) and one CAS loop: `ImageByteBudget::reserve` does `load` → check → `compare_exchange_weak` in a loop (`deepseek-recipe-image/src/limits.rs:158-178`), with `release` via `fetch_add` (`:181-183`). That is lock-free admission control against a byte budget. **Post-v1 relevance:** the same pattern admits requests against a KV-cache block budget.

**Pitfall to teach (M6)** `[inference, please verify by reasoning with Luigi]`: `Worker::new` (`utils/worker.rs:49-92`) waits on a **thread-local** `Mutex::new(())` (`:66`), while `stop_and_join` sets `stopping` and calls `notify_all()` **without holding any shared mutex** (`:94-99`). If the stop happens after the thread's `stopping` check but before it enters `condvar.wait` in the `Pause` state (`:71`), the notification is lost and `join()` can hang. That is the classic lost-wakeup race. `Wait(duration)` bounds it; `Pause` does not. `Worker` also has no `Drop`, so an un-stopped worker is detached. For M6, compare `std::thread::scope` (borrow data without `'static` and `Arc`, join guaranteed at scope end) against a persistent pool (llama.cpp style). The `F: FnMut() -> WorkerState + Send + 'static` bound (`:36`) is exactly what forces `Arc` clones in 3FS.

### 8.4 I/O `[code]`

- **Two file descriptors per file:** `normal_fd` opened with `O_SYNC`, and `direct_fd` with `O_DIRECT` when the filesystem supports it (`file/cluster.rs:17-41`). `pread`/`pwrite` choose the direct fd only when buffer address, length, and offset are all 4 KiB-aligned (`:63-105`).
- **Partial reads and EINTR:** loop on `read_at` until the buffer is full, treat `Ok(0)` as an error, and retry on `ErrorKind::Interrupted` (`:63-83`, `:107-113`). Same pattern for writes.
- `libc::fallocate` with `FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE` (`:9,43-61`) reserves or releases space.
- **No io_uring and no mmap** in the Rust parts (grep). The **USRBIO** C API wrapped by the `-sys` crate is io_uring-shaped: a registered shared-memory buffer (`Iov`) plus an I/O ring (`Ior`) with `prepare` → `submit` → `poll` of completions (`hf3fs-usrbio-sys/src/lib.rs:117-188`).
- **Map to M2 (mmap vs read):** the `read_at` loop shows what "read the whole file into a `Vec`" must handle; `std::fs::read` hides that loop. For mmap, candle uses `memmap2` in an `unsafe fn` whose `# Safety` says "The unsafe is inherited from `memmap2::MmapOptions`" (`candle-core/src/safetensors.rs:436-447`). mmap is `unsafe` because another process can change the file under your `&[u8]`. That is the SAFETY comment Luigi will have to write.

### 8.5 Benchmarks `[code]`

- **criterion** `benches/bench_allocator.rs:1-42`: set up a temp dir and allocator, then `c.bench_with_input(BenchmarkId::new("allocate", count), &count, |b, &c| b.iter(|| allocate(&allocator, c)))`. `[[bench]] harness = false` is in `Cargo.toml:40-42`.
- **Throughput binary** `src/bin/bench.rs`: N threads write for a fixed count while the main thread prints bytes/s once per second via `AtomicUsize::swap(0)` (`:42-80`).
- **Map to M6:** criterion for micro-kernels (matmul, rmsnorm); an end-to-end binary for tok/s, TTFT, and the M6 table. Neither 3FS bench pins CPU frequency or thread affinity, and CLAUDE.md asks for fixed prompt, fixed seed, and the command that produced the number.

---

## 9. Deep dive: FFI patterns

### 9.1 `cxx` (Rust ↔ C++), 3FS chunk_engine `[code]`

- `build.rs:1-4`: `let _ = cxx_build::bridge("src/cxx.rs"); println!("cargo:rerun-if-changed=src/cxx.rs");`. The returned `cc::Build` is **dropped without `.compile()`**, so cargo only generates the C++ half of the bridge. CMake compiles `target/cxxbridge/chunk_engine/src/cxx.rs.cc`, links `target/release/libchunk_engine.a`, and adds `target/cxxbridge` to the include path (`cmake/AddCrate.cmake:15-33`). C++ includes `"chunk_engine/src/cxx.rs.h"` (`src/storage/store/ChunkEngine.h:5`). `[code + inference on intent]`
- Bridge `#[::cxx::bridge(namespace = "hf3fs::chunk_engine")] pub mod ffi` (`cxx.rs:368-583`) contains shared POD structs visible to both languages (`UpdateReq`, `GetReq<'a>`, `RawMeta`, `RawUsedSize`, `FdAndOffset`, `Metrics`) and **only `extern "Rust"`** blocks. Rust exports opaque types (`Engine`, `Chunk`, `WritingChunk`, `RawChunks`, `LogGuard`) and methods. C++ never calls into Rust objects it does not own through these functions.
- **Ownership handoff:** `Box::into_raw(Box::new(engine))` returns `*mut Engine` (`cxx.rs:17`); `fn release(_engine: Box<Engine>) {}` takes it back and drops it (`:25`). `Arc::into_raw(c)` hands a refcount to C++ (`:101`), and `unsafe fn release_raw_chunk(&self, chunk: *const Chunk) { Arc::from_raw(chunk); }` gives it back (`:131-135`). C++ must pair every get with a release (`ChunkEngine.h:86,98`).
- **Map to M7:** Luigi's direction is the reverse (Rust host calls CUDA), so `cxx`'s `extern "C++"` or a plain C ABI would be the analogue. With **cudarc**, no hand-written FFI is needed: it wraps the driver API. The lesson that transfers is the **ownership handoff discipline**: who frees a device pointer, and when.

### 9.2 `bindgen` `-sys` crate, `hf3fs-usrbio-sys` `[code]`

- `build.rs:1-20`: `cargo::rustc-link-search=native={manifest}/lib` and `cargo::rustc-link-lib=hf3fs_api_shared` (`:6-7`); `bindgen::Builder::default().header(../../api/hf3fs_usrbio.h).clang_arg("-std=c99").parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))` (`:9-14`); write `OUT_DIR/bindings.rs` (`:16-19`).
- `lib.rs:1-5`: `#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]` + `include!(concat!(env!("OUT_DIR"), "/bindings.rs"));`, the standard `-sys` recipe.
- The `-sys` naming convention says "raw bindings only; safe wrappers in a separate non-sys crate". This crate **mixes** them: `Iov`, `Ior`, and `RegisteredFd` are in the same crate (`lib.rs:31-226`) `[inference: convention]`.
- **Soundness problems to use as teaching material** `[inference]`:
  - `unsafe impl Send for Iov {}` / `Sync` / `Send for Ior` (`:33-34,74`) with **no justification comment**.
  - `CString::new(mountpoint).expect(...).into_raw()` is passed to C and **never freed** (`:48-50,96-98`), a leak on every call.
  - `pub fn prepare<T>(..., extra: T)` stores `Box::into_raw(Box<PreparedIo<T>>)` as C userdata (`:117-139`). `pub fn poll<T>(...)` does `Box::from_raw(cqe.userdata as *mut PreparedIo<T>)` (`:153-183`). **Both are safe fns**, so calling `poll::<u32>` after `prepare::<String>` is undefined behaviour from safe code. A safe API must make UB impossible; this one does not. That is the most important FFI lesson here.
  - `impl Drop for Ior` / `RegisteredFd` (`:191-196,224-228`): RAII cleanup done right.
  - `test_io` (`:250-277`) needs a mounted 3FS at `/3fs/test`, so it cannot run in CI and is not `#[ignore]`d.
- **Map to M7:** candle-kernels compiles `.cu` → PTX in `build.rs` and `include!`s it (§2.3). Alternatives are nvcc → static lib + `cargo::rustc-link-lib`, or NVRTC at run time (cudarc has an `nvrtc` feature; candle enables it in `candle/Cargo.toml:48-59`). Use the `cargo::` (double-colon) syntax, as in usrbio-sys `build.rs:6-7` and deepseek-python `build.rs:6`, and emit `rerun-if-changed` for every `.cu`/`.cuh` (candle-kernels `build.rs:8-12`), or cargo will rebuild too often or not often enough.

### 9.3 PyO3 + maturin, `deepseek-recipe-python` `[code]`

- `Cargo.toml:9-12`: `[lib] crate-type = ["cdylib"], name = "_native", doc = false`; `pyo3 = { version = "0.27", features = ["abi3-py310"] }` (`:23`). abi3 builds one wheel for CPython ≥ 3.10.
- `pyproject.toml`: `build-backend = "maturin"`; `[tool.maturin] python-source = "python"`, `module-name = "deepseek_recipe._native"`, `features = ["pyo3/extension-module"]` (`:1-3,28-34` of pyproject, counted from its own line 1). The pure-Python `python/deepseek_recipe/__init__.py` re-exports from `._native`. **Hand-written** `_native.pyi` (581 lines) plus an empty `py.typed` give type checkers the API.
- `build.rs:1-8`: only a macOS linker flag, `cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names`, so `delocate` can rewrite library paths when making the wheel self-contained.
- `#[pymodule] fn _native(m)` registers constants (the special-token strings), exception types, and ~46 classes (`src/lib.rs:47-114`).
- Patterns: `#[pyclass(frozen)]` + inner `Mutex` or `Arc` (`tokenizer.rs:12-15`, `response.rs:282-285`); `py.detach(|| ...)` around CPU work so other Python threads run (`tokenizer.rs:58,65`; `encoding.rs:42,48`); a `macro_rules!` that generates identical bindings for V4 and V4.1 (`encoding.rs:14-66`); generators are **consumed** when passed to `StreamProcessor` (`Option::take`, `response.rs:351-377`), which mirrors Rust move semantics in Python.
- **Map to Luigi:** only if a side project wants Python bindings (e.g. calling his engine from the pytest reference harness). Not required by M0-M9.

---

## 10. Deep dive: testing

### 10.1 What exists `[code]`

| | deepseek-recipe | 3FS Rust |
|---|---|---|
| Location | Inline `#[cfg(test)] mod tests` in 5 files | Inline in 21 chunk_engine files, plus trash_cleaner and usrbio-sys |
| Count | 16 Rust tests (`#[test]` + `#[tokio::test]`) + 20 pytest (`test_bindings.py`) + 4 pytest (`server-py/tests/test_smoke.py`) | 46 (chunk_engine) + 3 (others) |
| `tests/` dir | none | none |
| Snapshot (insta) / property (proptest) | none | none |
| Async tests | `#[tokio::test]` (`resolver.rs:325+`) | n/a |
| Fixtures | bundled `tokenizer.json`; inline base64 PNG (`test_bindings.py:247-269`) | `tempfile::tempdir()` for real files and RocksDB (`cluster.rs:123`) |
| Test doubles | trait impls: `FixedFetcher`, `CountingFetcher` (records max concurrency with `AtomicUsize::fetch_max`), `NoPreprocessor`, `AcceptingPreprocessor` (`resolver.rs:249-301`) | real components on temp dirs |
| Table-driven golden values | `CASES` table of 36 `(w, h) → tokens/size` rows "Expected target sizes and token lengths of the V4.1 preprocessing" (`token_spec.rs:230-269`, test `:271-291`), and `fitted_images_are_valid` checks an invariant over the same table | n/a |
| Concurrency tests | CountingFetcher max-active | 16 threads × 1000 RocksDB batches (`rocksdb.rs:237-260`); commit vs get threads (`engine.rs:1689-1696`) |
| Opt-out tests | none | `#[ignore]` on the setuid test (`trash_cleaner/src/main.rs:506`) |
| Benchmarks | none | criterion (`benches/bench_allocator.rs`) |
| CI | none in repo; checks documented (`docs/development.md:66-71`, `:97-99`) | `cargo build --release` only (`.github/workflows/build.yml:28`); no `cargo test`/clippy/fmt |

### 10.2 What Luigi can copy (by milestone)

- **M1 kernel tests** (hand-computed values): the `token_spec.rs` `CASES` style is a `const` table of input → expected rows, a loop with `assert_eq!(..., "calc_resize({width}, {height})")` so a failure names its row (`:272-291`), and a second test that checks an **invariant** over the same rows (`:294-302`). For floats, swap `assert_eq!` for his D1 comparison `|ours - ref| ≤ atol + rtol·|ref|` and report the failing index (DECISIONS.md D1).
- **M3 golden-file test (logits vs reference):** neither repo has one. It would live in `tests/` (which needs a lib target, §3.1) or as an `#[ignore]`d unit test that reads the reference `.bin`/`.npy` produced by `scripts/tolerance/`, applies D1's atol = 5e-4 at every position, and prints PASS / near-tie / FAIL buckets. Fixture files: bundled small files (as deepseek bundles `tokenizer.json`) vs paths outside the repo (`~/refs/inference/models/`) are a decision about CI reproducibility.
- **M4 streaming decode:** port deepseek's Python cases into Rust unit tests. The emoji split across byte tokens gives the right text **and** counts every token (`test_bindings.py:163-188`). An unknown id is handled (`:191-213`). A stop sequence split across chunks (the example in §6.3).
- **M6 threads:** 3FS-style stress (N threads × M iterations) and, for kernels, **multi-threaded result == single-threaded result** (bitwise, or within tolerance if the reduction order changes, which is itself something to discuss).
- **M7 GPU tests:** `#[ignore]` (3FS style) or `#[cfg(feature = "cuda")]` so `cargo test` passes without a GPU. Every GPU kernel is compared to the CPU oracle with a measured tolerance (D1 method).
- **Test doubles for M7 or post-v1:** deepseek's traits with default "unsupported" bodies make fake backends one line long (`impl ImagePreprocessor for NoPreprocessor {}`), useful for a fake engine in server tests (the mock inference in `server-rs/src/lib.rs:315-333`).

---

## 11. Deep dive: tooling configuration

| Item | deepseek-recipe | 3FS | candle |
|---|---|---|---|
| Toolchain pin | `rust-toolchain.toml`: `channel = "1.97.1"`, components rustfmt/clippy/rust-src/rust-analyzer (`:1-4`) | none (CI installs distro `rustc cargo`, `build.yml:20`) | none |
| MSRV (`rust-version`) | none | `1.85.0` (`Cargo.toml:17`) | none |
| Edition | 2024 | 2021 | 2021 |
| rustfmt | `rustfmt.toml`: `style_edition = "2024"` only (comment: "Use the same style edition in Cargo and standalone rustfmt runs") | none (only `.clang-format` for C++) | none |
| clippy config / lints | none; `-D warnings` via command | none | `-D warnings` in CI and pre-commit |
| rustdoc | `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked` (`docs/development.md:70`) | none | n/a |
| `--locked` | every documented command | no | no |
| deny.toml / pre-commit | none | none | `.pre-commit-config.yaml` (fmt, clippy `--tests --examples -- -Dwarnings`) |
| `.cargo/config.toml` | none | none | `target-cpu=native` |

Notes:
- Luigi has Rust **1.96**; deepseek pins **1.97.1**. Toolchain pinning is itself an M0 option: `rust-toolchain.toml` makes benchmarks reproducible (compiler version affects codegen) at the cost of a rustup download.
- `--locked` makes `cargo` fail instead of silently updating `Cargo.lock`. That matters for "reproducible from one command" (v1.0 done-condition).
- `target-cpu=native` (candle) vs `-C target-feature=+avx2,+fma` vs runtime detection is an M6 decision. It changes what the binary runs on, and whether `#[target_feature(enable = "avx2")]` functions need `unsafe` call sites.
- **No repo uses `[lints]`.** CLAUDE.md's SAFETY rule can be enforced with `clippy::undocumented_unsafe_blocks` (§3.5). That is a question for Luigi, not something to add.

---

## 12. Rust idioms cheat sheet (from these codebases)

Each entry gives: where, when Luigi needs it, and the borrow-checker angle.

1. **Consuming builder with `#[must_use]`** (`options.rs:45-50`). `fn with_x(mut self, x) -> Self`. *M4* sampler/generate options. Borrow: taking `self` by value avoids `&mut` chains; `#[must_use]` catches `opts.with_x(1);` whose result is discarded.
2. **`#[non_exhaustive]` on public structs and enums** (`options.rs:4,15`). *M4/M8* public config types. Borrow: none. API: outside crates must use constructors, and `match` needs `_`.
3. **Optional capability as `Option<Box<dyn Trait>>` attached with `impl Trait + 'static`** (`dsv41.rs:19,31-34`; `processor.rs:18,41-44`). *M4* tokenizer or sampler plug-ins. Borrow: `'static` means "owns its data", so the struct needs no lifetime parameter. The cost is dynamic dispatch (fine outside hot loops).
4. **Blanket impl for `Arc<T>`** (`decoder.rs:28-35`, `tokenizer.rs:24-28`). *M6/M7* sharing weights or the tokenizer across threads. Borrow: `(**self).method()` derefs through `&Arc<T>`. The `T: Sync` bound is required for `Arc<T>: Send`.
5. **Internal hook trait + public blanket impl** (`v4/mod.rs:71,206`). *M5* quant formats, *M7* backends. Borrow: none. Design: implementors supply constants and small hooks, and the shared algorithm is written once.
6. **let-else** (`processor.rs:106-109`). *M2* header parsing: `let Some(x) = ... else { return Err(...) };`. Borrow: the binding lives in the outer scope with no extra nesting.
7. **Let chains `if a && let Some(x) = f()`** (`processor.rs:336-337`). Edition 2024, Rust ≥ 1.88 (Luigi's 1.96 is fine). *Any*. Borrow: a temporary in the condition lives through the block. Watch for held `RefCell`/`Mutex` guards.
8. **`std::mem::take` to move out of a `&mut` field** (`decoder.rs:69`, `processor.rs:252`). *M4* streaming buffers: hand out the accumulated `String`/`Vec` and leave an empty one. Borrow: this is the standard fix for "cannot move out of `self.x` which is behind a mutable reference".
9. **Return byte-length segments, not borrowed slices** (`state_machine.rs:121-125`). *M4* incremental parsers and tokenizers. Borrow: the parser holds no reference into caller data across calls, so it can live in a long-lived struct without lifetime parameters.
10. **Replace the whole state on a transition: `*self = State::new(next, opts)`** (`state_machine.rs:412`). *M4* generate-loop phases (prefill → decode → done). Borrow: assigning to `*self` inside `&mut self` is fine once no other borrow of `self` is live (the matched values were copied out first at `:394-398`).
11. **KMP incremental matcher over bytes** (`state_machine.rs:481-527`). *M4* stop strings. Borrow: none. Correctness: the failure table handles overlap ("aab" in "aaab").
12. **Aligned allocation with `std::alloc::Layout` + `Drop` + `Deref<Target = [T]>`** (`aligned.rs:4-39`, counter-example). *M1* buffer type. Borrow: `Deref` gives `&[T]` and `&mut [T]` for free. Do **not** store `&'static mut`; use `NonNull<T>` + `len` + `PhantomData`.
13. **`thread_local!` + `RefCell` scratch buffer** (`chunk.rs:25-27,77-83`). *M6* per-thread matmul or attention scratch. Borrow: `RefCell::borrow_mut` is checked at run time, and a re-entrant call on the same thread panics.
14. **RAII `Drop` for external resources** (`chunk.rs:302-306`; `usrbio lib.rs:191-196,224-228`; trash_cleaner `UserContext` `main.rs:358,377`). *M2* mmap, *M7* device buffers and streams. Borrow: `Drop` takes `&mut self`, so moving fields out of `self` inside it needs `Option::take` or `mem::take`.
15. **`#[repr(C)]` + compile-time layout asserts** (`chunk_meta.rs:6`, `cxx.rs:585-604`). *M2* checkpoint header struct, *M7* kernel parameter structs passed to CUDA. Borrow: none. Soundness: `repr(Rust)` field order is unspecified, so never cast bytes to a non-`repr(C)` struct.
16. **Newtype with `const fn` constructors** (`size.rs:1-37`). *M1* dimensions and byte sizes, *M5* block sizes. Borrow: `Copy` newtypes act like integers. Pitfall: lossy `From` (`size.rs:79`).
17. **Const generics + array init** `ShardsMap<K, V, const S: usize = 64>`, `[(); S].map(|_| Default::default())` (`shards_map.rs:8,23`). *M5* `QK = 32` block types. Borrow: none. Arrays of non-`Copy` types cannot use `[x; S]`, hence the `map` trick (or `std::array::from_fn`).
18. **CAS loop on an atomic** (`limits.rs:158-178`). *post-v1* KV-block admission. Borrow: atomics give shared mutation through `&self`, with no `&mut`.
19. **Thread with `FnMut() -> State + Send + 'static` closure + stop flag** (`worker.rs:34-99`). *M6* workers. Borrow: `'static` forces `Arc` clones of everything captured. `std::thread::scope` removes that requirement for fork-join kernels.
20. **`self: &Arc<Self>` receiver** (`allocator.rs:24`). *M7* device or context handles that create child objects keeping the parent alive. Borrow: lets a method `clone()` the `Arc` into the child.
21. **Generated code via `include!(concat!(env!("OUT_DIR"), "/x.rs"))`** (usrbio `lib.rs:5`; candle-kernels `lib.rs:1-3`). *M7* PTX embedding. Borrow: none. Build: pair it with `cargo::rerun-if-changed`.
22. **Feature-gated module + re-export** (`deepseek-recipe-image/src/lib.rs:17-30`), or a dummy module under the same name (candle `lib.rs:113-117`). *M7* `cuda` feature. Borrow: none. API: the dummy-module pattern keeps enums and signatures identical across feature sets.
23. **`async_stream::stream!` + `std::pin::pin!`** (`processor.rs:65-66`). *post-v1* SSE token streams. Borrow: `pin!` pins on the stack. Values held across `.await` must be `Send` for multi-threaded runtimes (hence `G::Chunk: Send`, `:59-60`).
24. **Read loop handling short reads and EINTR** (`cluster.rs:63-83,107-113`). *M2* if Luigi uses `read` instead of mmap. Borrow: `buf = &mut buf[n..]` reborrows the remaining slice, a neat `&mut` slice idiom.

---

## 13. Milestone map

| Milestone | What from these repos | Refs |
|---|---|---|
| **M0** crate layout | virtual workspace vs package; lib+bin (`server-rs`, `chunk_engine`); `[workspace.dependencies]`; `default-members`/`exclude`; profiles at root; toolchain pin; `--locked` | §2, §3, §11 |
| **M1** tensor/buffer + kernels | `AlignedBuffer` critique; `repr(align)` alternative; newtype; table-driven tests with row labels; panic vs Result for shapes | §4.4, §5.4, §8.1, §10.2 |
| **M2** loading (mmap vs read) | `read_at` loop, O_DIRECT alignment rules, candle's `unsafe fn` mmap + alignment check, `repr(C)` header, endianness of transmutes, weights at byte 28 | §8.1, §8.2, §8.4 |
| **M3** forward + golden test | golden-file test needs lib target or `#[ignore]`; D1 buckets | §3.1, §10.2 |
| **M4** tokenizer, streaming, CLI | `StreamDecoder` (ids → text on complete character), byte-buffer toy, vLLM offsets, llama2.c `safe_printf` loss, KMP stop strings, stashing latency, tokenizer trait, BOS ownership, push vs pull | §6, §7 |
| **M5** quantization | hook trait + blanket impl for formats; const-generic block sizes | §5.2, §12 #5, #17 |
| **M6** SIMD + threads | per-thread scratch; lost-wakeup pitfall; `thread::scope` vs `'static` workers; stress tests; criterion; `debug = true` profile; LTO/inlining across crates; `target-cpu` | §3.1, §8.3, §8.5, §11 |
| **M7** CUDA | candle `exclude` + feature + dummy backend; `build.rs` .cu → PTX + `include!`; RAII device buffers; ownership handoff (cxx `Box`/`Arc` into/from raw); `-sys` soundness lessons; `#[ignore]`/feature-gated GPU tests | §2.3, §9, §10.2 |
| **M8** real model | HF `tokenizers` crate, `tokenizer.json` added tokens, chat template in code vs Jinja, injection surface, token-id constants | §7 |
| **post-v1** server | `InferenceChunk` contract; SSE framing + `[DONE]`; `AppendDelta` (one stream → streaming or complete response); error → HTTP mapping; `is_retryable`; CAS admission; tracing JSON event layer (`trash_cleaner/src/main.rs:656-676`) for p99 analysis | §4, §5.3, §6.5, §8.3 |

---

## 14. Quiz questions (explain-back)

1. deepseek-recipe's `StreamDecoder` waits while decoded text ends with U+FFFD. Why is that correct for DeepSeek's tokenizer, and which two things break if you use the same trick with a SentencePiece tokenizer? (Expected: byte-level BPE vs neighbour-dependent space stripping; legitimate U+FFFD output; stuck undecodable ids.)
2. In the §6.3 example, chunk 4 produced a `Raw 13` segment but the client saw nothing until chunk 5. Walk through `apply_actions` and say why. Is that a bug or a trade-off?
3. Why is `String::split_off(len)` in `StashedChunks::pop` safe even though the state machine counts **bytes**? What property of UTF-8 does it rely on?
4. You want `cargo test` to pass on a laptop without nvcc after M7. Name two different mechanisms from these repos and one downside of each.
5. Your M3 golden test lives in `tests/logits_vs_reference.rs`. What must be true about your crate layout for it to compile?
6. `AlignedBuffer::new` is a safe function. Give a line of safe code that causes undefined behaviour with it, and the one-line fix.
7. 3FS marks `speed_up_quit` as `unsafe fn` although it only leaks memory. What does `unsafe` promise in Rust, and does this function need it?
8. `hf3fs-usrbio-sys::Ior::poll<T>` is safe. Show how a caller triggers UB, and propose an API change that prevents it.
9. `thiserror` in `-image` and `-encoding`, hand-written errors in `-core` and `deepseek-recipe`: why might a team split it that way? What would you use in M2, and why?
10. The `Worker` in 3FS can hang in `stop_and_join`. Describe the interleaving. How does `std::thread::scope` avoid the whole class of problem for a fork-join matmul?
11. The llama2.c checkpoint's weights start at byte 28. What does that imply if you mmap the file and use `_mm256_load_ps` on the weights directly?
12. deepseek renders `<｜begin▁of▁sentence｜>` as **text** and tokenizes with `add_special_tokens = false`. llama2.c prepends BOS as an **id**. What goes wrong in M8 if both happen?
13. Why must `profile.release` settings live in the workspace root and not in `crates/engine/Cargo.toml`?

---

## 15. Open questions (not verified; check before relying on them)

1. **deepseek-recipe CI:** no `.github/` in the public tree. Whether DeepSeek runs the documented checks internally is unknown. The public repo is a single squashed commit.
2. **Cross-crate inlining specifics** (automatic inlining of small non-generic functions, from about Rust 1.75): verify on 1.96 with `cargo asm` or by benchmarking if Luigi splits kernels into their own crate.
3. **`EncodingV4` is `pub(crate)` but bounds a `pub` blanket impl** (`v4/mod.rs:71,206`). It compiles under their `clippy -D warnings`. Whether the `private_bounds` lint ever fires here was not tested.
4. **HF Llama/TinyLlama `tokenizer.json` decoder behaviour** (leading-space `Strip` when decoding one token in isolation): stated from memory as the reason vLLM keeps prefix offsets. Verify on the actual M8 tokenizer file.
5. **Special-token injection** in deepseek-recipe (user text containing `<｜Assistant｜>`). This is inferred from the render-then-tokenize design and from `special: false` added tokens. Not tested.
6. **3FS `Worker` lost wakeup:** inferred from the code structure, not reproduced.
7. **Uninitialized reads through `AlignedBuffer`:** every call site I read writes before reading (`fill(0)`, `pread`). The unsoundness is in the API, not an observed bug.
8. **cudarc API names** (context, stream, device slice types) were not read in this pass. Re-check against the cudarc version candle pins (`0.19.10`, `candle/Cargo.toml:48`) when M7 starts.
9. **Unused `release-cmake` profile in 3FS:** grep found no users in the repo. External build scripts (Docker images) might use it.
