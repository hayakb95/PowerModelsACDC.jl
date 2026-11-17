# GPU execution for HVDC OPF

PowerModelsACDC's continuous nonlinear AC/DC OPF models can use MadNLP's GPU
solver through the existing JuMP optimizer interface. The AC network, DC network,
and converter equations are built by the same PowerModelsACDC functions used for
a CPU solve.

This configuration stores MadNLP's solver vectors and the cuDSS factorization on
an NVIDIA GPU. JuMP evaluates the model functions and derivatives on the CPU and
MadNLP transfers those evaluations to its GPU solver. It does not require rewriting
the converter equations. Performance speedups are not a requirement for this
example.

For GPU evaluation of the model functions and derivatives as well, see
[ExaModels evaluation for HVDC OPF](@ref). That workflow has a separate optional
environment and compares ExaModels CPU/CUDA evaluations with the original JuMP
expressions.

## Optional environment

The example has its own environment; CUDA packages are not required by the main
PowerModelsACDC package or its ordinary CPU test suite. From the repository root:

```sh
julia --project=tutorials/gpu -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=tutorials/gpu tutorials/gpu/opf.jl
```

Run the GPU command on a node with an allocated NVIDIA GPU and a working CUDA
driver. Both `CUDA` and `CUDSS` must be loaded to activate MadNLPGPU's CUDA
extension. On a cluster, launch the Julia process inside the allocation and set
`JULIA_NUM_THREADS=1` when only one CPU is assigned.

The default example is `case5_acdc.m`, containing five AC buses, three DC buses,
three VSC converters, and a meshed DC network. Supply an extended MATPOWER case
to solve another HVDC model:

```sh
julia --project=tutorials/gpu tutorials/gpu/opf.jl test/data/case5_lcc.m
julia --project=tutorials/gpu tutorials/gpu/opf.jl --cpu
```

## Optimizer configuration

The essential GPU configuration is:

```julia
using JuMP, PowerModels, PowerModelsACDC
using CUDA, CUDSS, MadNLP, MadNLPGPU

CUDA.functional() || error("CUDA is not functional")
CUDA.allowscalar(false)

optimizer = optimizer_with_attributes(
    MadNLP.Optimizer,
    "array_type" => CUDA.CuArray,
    "linear_solver" => MadNLPGPU.CUDSSSolver,
    "tol" => 1e-8,
    "acceptable_tol" => 1e-8,
    "bound_relax_factor" => 1e-8,
)
data = PowerModelsACDC.parse_file("test/data/case5_acdc.m")
result = PowerModelsACDC.solve_acdcopf(data, PowerModels.ACPPowerModel, optimizer)
result["solution"]["busdc"]
result["solution"]["branchdc"]
result["solution"]["convdc"]
```

The complete example also makes the input starting point explicit for paired
CPU/GPU checks and returns the instantiated PowerModels model for validation.

## Numerical checks

MadNLP's default GPU condensed system relaxes equality constraints and uses
different defaults from its ordinary CPU path. Setting `tol` alone does not
establish original-model feasibility. Keep the relaxation explicit and check the
original JuMP constraints and bounds after solving:

```julia
using CUDA, CUDSS, MadNLPGPU
include("tutorials/gpu/opf.jl")
result, pm = solve_hvdc_example()
violations = JuMP.primal_feasibility_report(pm.model; atol=0.0)
maximum(values(violations); init=0.0)
```

The example requires a feasible solution status and maximum original-model
violation at most `1e-6`. The GPU regression entry point compares fresh CPU and
GPU solutions for the VSC and LCC cases, checks objective agreement at relative
tolerance `1e-6`, and verifies GPU solver storage:

```sh
julia --project=tutorials/gpu tutorials/gpu/runtests.jl
```

This explicit GPU suite fails when CUDA is unavailable; a skipped test should
not be interpreted as GPU validation. The same numerical checks apply even if
the GPU execution takes longer than the CPU reference. Nonlinear OPF solutions
are local optima, and larger cases may converge to different local solutions.

## CPU-only comparison tests

The optional environment also contains tests of the comparison logic that do
not load CUDA or require a GPU. Run these on a CPU node after instantiating the
same environment:

```sh
julia --project=tutorials/gpu tutorials/gpu/validation/runtests.jl
julia --project=tutorials/gpu tutorials/gpu/validation/test_records.jl
```

The first uses small synthetic JuMP models to test valid redistributions and
rejection of individual limit violations, wrong bus totals, different/curved
costs, additional coupling, changed AC/DC/converter fields, inconsistent saved
values, failed statuses and nonfinite values. The second checks CLI acceptance
selection and saving/rebuilding/rechecking one small CPU solution. Its record
round trip does not constitute a GPU test. Neither entry point changes the
ordinary package test target or requires a GPU to be present.

## Portable paired experiments

The optional launcher uses the same case, explicit starts, and matched
`RelaxEquality`/`RelaxBound` numerical treatments for CPU and GPU. It preserves
each solution and reports both comparison outcomes. Run on an allocated GPU
node, from the repository root:

```sh
# Small meshed VSC and LCC cases; strict comparison is the default.
julia --project=tutorials/gpu tutorials/gpu/validation/run.jl \
    --output tutorials/gpu/results/small

# Explicit larger-case experiment: 39, 588 and 3120 AC buses.
julia --project=tutorials/gpu tutorials/gpu/validation/run.jl \
    --large --comparison verified-sharing --output tutorials/gpu/results/large

# Or choose bundled case names or paths to extended MATPOWER case files.
julia --project=tutorials/gpu tutorials/gpu/validation/run.jl \
    --comparison verified-sharing --output tutorials/gpu/results/case39 case39_acdc.m
```

The launcher has no scheduler or machine-specific commands. Arrange CPU/GPU
resources externally; use one Julia/BLAS/OpenMP thread when assigned one CPU.
The CPU reference uses MUMPS and the GPU run uses cuDSS, with JuMP evaluation on
the CPU. Solves are sequential and have a 300-second solver wall-time limit
each. Compilation and parsing can add elapsed time. A new output directory is
required; existing results are never overwritten. Results under
`tutorials/gpu/results/` are ignored by Git. Cases are read from the installed
PowerModelsACDC package or from explicit caller paths.

Each case directory contains `case.m`, `cpu.toml`, `gpu.toml`, strict
`comparison.toml`, `sharing_comparison.toml`, its rebuilt canonical model, and
`outcome.toml`. `summary.toml` records the whole experiment. A failed solve or
selected comparison returns a nonzero process exit code. The active project
and resolved manifest (when present) are copied into the output directory for
reproduction. An exception also
fails the process and leaves partial evidence; absence of a completed summary
must not be interpreted as success. The selected comparison cannot override a
failed CPU or GPU solve.

## Verified generator sharing

Strict comparison checks every generator output individually. A continuous
OPF can admit several allocations among generators connected to one bus. For
example, the objective need not select a particular reactive-power split.
The optional `--comparison verified-sharing` mode permits only redistributions
that the rebuilt JuMP model structurally supports:

* Generators have the same AC bus and the same output field (`pg` or `qg`).
* Their variables enter exactly one non-bound equality with identical unit
  coefficients, without nonlinear or quadratic occurrences.
* Their linear objective coefficients are identical; for `qg` they must be zero.
* Fixed generator variables are excluded. All individual original bounds and
  every original model constraint remain checked for each saved solution.

Unsupported generator dependencies retain individual comparisons. Unknown
expression types are refused. The current implementation supports only
continuous, single-network ACP OPF with the example's base settings; it does
not enable this treatment for ACR, unit commitment or multinetwork models.

Verified group totals use the same `1e-4` absolute physical tolerance, without
multiplying by group size. Other physical quantities retain individual checks.
Original feasibility must be within `1e-6`, and relative objective agreement
within `1e-6`. Saved objectives must match model re-evaluation (`1e-10` relative
or `1e-8` absolute). The comparator matches case/model/start fingerprints,
dimensions and model-building package versions, and binds the saved physical
output fields to the complete primal point it rechecks.

A report can therefore say `strict_passed = false` and `passed = true` in the
sharing mode. Both outcomes and all individual differences are retained. This
is agreement under explicitly verified sharing freedoms, not a proof of global
optimality. Strict remains the default experiment acceptance mode.

To compare saved runs later, on a CPU node with the matching optional environment:

```sh
julia --project=tutorials/gpu tutorials/gpu/validation/compare.jl \
    SAVED_CASE/cpu.toml SAVED_CASE/gpu.toml NEW_STRICT_REPORT.toml
julia --project=tutorials/gpu tutorials/gpu/validation/sharing.jl \
    SAVED_CASE/cpu.toml SAVED_CASE/gpu.toml SAVED_CASE/case.m NEW_SHARING_REPORT.toml
```

The sharing postprocessor rebuilds an unsolved model and checks saved points;
it does not invoke an optimizer or load CUDA. Retain the resolved environment
manifest with the results for reproducible replays. A fingerprint or version
mismatch is a failure. The comparison helper is an optional validation tool,
not a new exported package API.

This example covers continuous nonlinear OPF. It does not provide a mixed-integer
GPU solver for unit commitment or transmission expansion planning, or a GPU
implementation of the sequential NLsolve power-flow algorithm.

For details of the GPU wrapper and condensed-system numerical choices, see the
[MadNLP GPU documentation](https://madsuite.org/MadNLP.jl/stable/tutorials/gpu/).
