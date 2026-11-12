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

This example covers continuous nonlinear OPF. It does not provide a mixed-integer
GPU solver for unit commitment or transmission expansion planning, or a GPU
implementation of the sequential NLsolve power-flow algorithm.

For details of the GPU wrapper and condensed-system numerical choices, see the
[MadNLP GPU documentation](https://madsuite.org/MadNLP.jl/stable/tutorials/gpu/).
