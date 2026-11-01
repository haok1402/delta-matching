**Verdict: SAFE at the source/PTX schedule level.** H1b preserves the group ledger and can remain bitwise. Fitting within 240 registers without spills or compiler-inserted serialization remains unproven. PROF/TRACE accounting needs adjustment.

I only read files and documentation; I made no edits and ran no GPU work. Line references below describe the current checkout.

1. **WGMMA legality and operand fences**

   PTX forbids ordinary accesses to an outstanding WGMMA’s accumulator or register-A operands before the corresponding group wait. This includes **reads**, even if the value is unchanged. [PTX `wgmma.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).

   H1b satisfies that rule **provided the deferred transpose reads only the eight independent packed words**:

   | Registers | Last ordinary access before issue | First permitted ordinary access afterward |
   |---|---|---|
   | `rP` | Conversion | After dV’s `wait<1>` |
   | `tdVrdV` | Window rescale | After dV’s `wait<1>`; promotion remains after `wait<0>` |
   | `rdS` | Pre-pack byte permutations | After dK’s `wait<0>` |
   | `fq8_tmpK` | Existing GEMM setup | Fold after dK’s `wait<0>` |

   The current conversion, transpose, rescale, issue and fold locations are [mainloop:871–960](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:871). Moving the existing `rdS32` reads at lines 893–894 after dK issue would be illegal; moving only their already-computed results is legal.

   **The helper’s existing fence scheme is sufficient for this source schedule.** It fences RS A and C, executes `warpgroup_arrive()`, issues the K loop, commits, and applies trailing operand fences. With `wg_wait=-1`, it performs no internal wait. See [utils.h:270–320](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/third_party/flash-attention/hopper/utils.h:270).

   Distinguish the two mechanisms:

   - `warpgroup_fence_operand` is empty inline assembly with read/write register constraints—a compiler dependency.
   - `warpgroup_arrive()` emits the actual `wgmma.fence.sync.aligned`.

   See [GMMA primitives:49–101](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/third_party/flash-attention/csrc/cutlass/include/cute/arch/mma_sm90_gmma.hpp:49). FP8 fragments are fenced as packed `uint32_t` words by [the tensor overload:49–64](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/third_party/flash-attention/csrc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:49).

   No extra hardware fence is needed for the packed copies: they are ordinary registers, never WGMMA operands. The existing hardware fence orders prior A/C accesses before issue. [PTX `wgmma.fence`](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence).

   **Named barriers, shuffles, `stmatrix`, the proxy fence and TMA store are legal with groups outstanding.** They neither access the protected registers nor overwrite either GEMM’s shared-memory operand. The storage fields are separate ([storage:275–285](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:275)); dV/dK read the dOt/Qt fragments ([partitioning:640–641](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:640)). Preserve the existing full-warp shuffle masks and WG-uniform barrier participation.

2. **Group ledger and shared-memory ownership**

   At this point S and dP have already been drained by the earlier waits ([mainloop:757–775](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:757)).

   | Point | Committed groups not yet guaranteed retired |
   |---|---|
   | Before dV | None |
   | After dV helper | dV |
   | After dK helper | dV, dK |
   | After deferred transpose/store | dV, dK |
   | After `wait<1>` | At most dK |
   | After `wait<0>` | None |

   Each helper call commits once. TMA’s bulk-group commit is a separate mechanism and does not change this ledger. Thus dOt release remains after dV retirement, Qt release after dK retirement, and the fold remains after both: [mainloop:929–960](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:929). These are completion guarantees; hardware may finish either operation earlier.

   The `sdS` reuse sequence remains correct:

   `issuer tma_store_wait<0>` → WG barrier → both `stmatrix` → every writer’s proxy fence → WG barrier → issuer TMA store/commit.

   Preserve [lines 903–912](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:903) together. The issuer is consistently `wg_thread==0`, and its store helper commits immediately ([703–709](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:703)).

   **Precision:** `tma_store_wait<0>()` emits `cp.async.bulk.wait_group.read 0`, so it drains prior **source reads**, which is sufficient to overwrite `sdS`; it does not promise complete global-store visibility. [Implementation:1245–1254](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/third_party/flash-attention/csrc/cutlass/include/cute/arch/copy_sm90_tma.hpp:1245), [PTX bulk wait](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group). The final store wait remains at [mainloop:988–989](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:988).

3. **240-register budget**

   Conservatively reserving outstanding A operands through their waits:

   | Values | 32-bit registers/thread |
   |---|---:|
   | `tdVrdV` | 64 |
   | `tdKrdK` | 64 |
   | `fq8_tmpK` | 64 |
   | `rdS` | 8 |
   | `rP` | 8 |
   | Packed transpose words | 8 |
   | **Subtotal** | **216** |

   The 64-element accumulators are asserted at [line 660](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:660); `fq8_tmpK` matches dK ([714](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:714)); the RS atom uses four A words per K32 operation and 64 FP32 accumulators ([GMMA atom:14174–14179](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/third_party/flash-attention/csrc/cutlass/include/cute/arch/mma_sm90_gmma.hpp:14174)).

   That leaves **24 registers before shuffle scratch, addresses, scales, pipeline state and loop/scheduler state**. Two sequential shuffle temporaries bring the subtotal to roughly 218. The surrounding state is substantial: [scales and addresses:668–710](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:668).

   My estimate is **near the 240 limit, plausibly exceeding it before rematerialization and uniform-register allocation**. A carefully scalarized default build may fit; spills or earlier retirement are a credible risk, especially under PROF/TRACE. Neither is inevitable from the fragment count alone.

   To keep staging at eight words:

   - Pre-pack directly into one fixed `uint32_t packed[8]`.
   - Fully unroll all accesses.
   - In the deferred block, consume each word’s two shuffles, then overwrite that same element with its final byte permutation.
   - Process one word at a time to limit shuffle scratch.
   - Avoid retaining separate packed, shuffled and output arrays; avoid `volatile` or address escape.

   This lets completed output words replace dead packed inputs instead of growing an additional eight-word live set.

   **Do not interpret the recorded `REG:168` as 72 spare consumer registers.** It is consistent with the redistributed allocation: `(24 + 240 + 240)/3 = 168`. The runtime allocation is explicit at [kernel:165,187](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/flash_bwd_kernel_sm90_fp8r.h:165). Existing resource records describe the old schedule, including STACK 8/0 for non-causal/causal ([resource report:6–10](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/results/r4/resource_pass1_do2.txt:6)).

   Source review cannot guarantee physical register non-reuse or preserved overlap. H1b’s eventual PTX/SASS must confirm pre-pack placement, no protected-register accesses before retirement, no added serialization, and spill behavior.

4. **Causal, non-causal and instrumentation paths**

   Both masked and unmasked loops invoke the same lambda with runtime staging: [mainloop:974–986](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:974). `Is_causal` changes masking/bounds, not this issue/retirement sequence. The reasoning therefore covers both causal loops and the non-causal instantiation.

   DBG reads S/dP and completes its reductions before conversion/issue ([779–838](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/mainloop_bwd_sm90_fp8r.hpp:779)). Leave that block there.

   Default KP/KT macros compile away. Enabled probes access clocks and instrumentation buffers, not WGMMA operands ([kda_prof.h:12–27,66–88](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/kda_prof.h:12)). Their extra state still warrants separate register inspection. PROF+TRACE and DBG+profiling combinations are explicitly rejected ([trace guard](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/kda_prof.h:38), [DBG guard](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/csrc/delta_matching_api.cpp:224)).

   **Reporting change needed:** keep issue markers immediately after their GEMMs and transpose/store markers with their work. Then update accounting: `ev[16]-ev[15]` will include the deferred transpose/store, and `windows()` currently omits that new vector interval. See [trace accounting:96–129](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/benchmarks/trace_pass1.py:96). Dense PROF’s successive-clock accounting likewise needs its phase labels reviewed ([phase map:32–38](/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/hcai/workspace/haok1402/delta-matching/benchmarks/profile_pass1.py:32)).

5. **Bitwise behavior**

   **No arithmetic reason prevents bitwise equality.** The conversion inputs, selectors, shuffle sources, GEMM K order, rescale-before-dV relationship, zero-init dK temporary, fold order and promotion remain unchanged. The moved operations only rearrange already-produced bytes and publish a private shared-memory tile.

   This is a bitwise-preserving schedule argument, not an empirical certification. The unresolved issue is generated-code register pressure and whether the intended overlap survives compilation.
