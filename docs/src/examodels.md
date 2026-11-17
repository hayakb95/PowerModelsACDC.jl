# ExaModels evaluation for HVDC OPF

The optional ExaModels workflow converts the existing single-network,
continuous ACP AC/DC OPF model to an ExaModels evaluator. With the CUDA backend,
ExaModels evaluates the objective, constraints, gradient, Jacobian, and Hessian
on the GPU, and MadNLPGPU uses cuDSS for sparse factorization. Parsing, JuMP
model construction, conversion/setup, and independent validation use the CPU.
The standard PowerModels solution dictionary includes the AC, DC, and converter
results. Speedups are not an acceptance requirement.

This is a separate optional environment from the [JuMP GPU solver example](@ref
"GPU execution for HVDC OPF"). The main package has no ExaModels or CUDA
dependency. Other formulations, multinetwork problems, and integer decisions
are outside this example's validated scope.

## Setup and execution

From the repository root:

```sh
julia --project=tutorials/examodels -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=tutorials/examodels tutorials/examodels/run.jl --output tutorials/examodels/results/small
```

Run the second command inside an NVIDIA GPU allocation. Load the CUDA driver
through the cluster's normal environment and set `JULIA_NUM_THREADS=1` if the
allocation supplies one CPU. The command fails when CUDA is unavailable; it
does not substitute a CPU run. The output directory must not already exist.

The default cases are `case5_acdc.m` (meshed VSC) and `case5_lcc.m`. Each case
runs with matched starting points and MadNLP numerical treatments through:

1. JuMP evaluation and MadNLP/MUMPS on the CPU.
2. ExaModels evaluation and MadNLP/MUMPS on the CPU.
3. ExaModels CUDA evaluation and MadNLPGPU/cuDSS on the GPU.

To check conversion without loading any CUDA package:

```sh
julia --project=tutorials/examodels tutorials/examodels/run.jl --cpu-only --output tutorials/examodels/results/cpu
```

The summary explicitly records whether GPU execution was validated. A CPU-only
pass is not GPU validation. To select another bundled or external case, append
its filename to the command. To run the bundled 39, 588, and 3120 bus cases:

```sh
julia --project=tutorials/examodels tutorials/examodels/run.jl --large --comparison verified-sharing --output tutorials/examodels/results/large
```

## Acceptance checks and saved evidence

Every solve must report native MadNLP success and a feasible MOI status. Its
complete primal point is replayed against the original JuMP model, including
all converter and DC equations, with maximum constraint and bound violations
at most `1e-6`. Objective differences must be at most `1e-6` relative, and
strict physical-field differences at most `1e-4` absolute.

The workflow compares ExaModels with independent MOI sparse reverse AD built
from the original JuMP expressions, using the actual optimizer variable and
constraint maps. It checks variable/constraint bounds, objective and constraint
values, gradients, Jacobians, and Hessians at the starting point, a bounded
perturbation, and the returned solution. Hessians include objective-only,
constraint-only, and combined Lagrangian curvature with nonzero constraint
multipliers. Sparse duplicate coordinates are summed before comparison.
Each entry must satisfy `abs(a-b) <= 1e-7 + 1e-8*max(abs(a),abs(b))`.

GPU checks require CUDA model/solver arrays, CUDA evaluation buffers, a GPU
factorization matrix, and successful cuDSS status with scalar indexing disabled.
These checks establish execution and numerical agreement for the tested models;
they do not measure kernel performance or imply that host setup runs on the GPU.

Every pair retains a strict report. The explicitly selected
`--comparison verified-sharing` mode also uses the existing structural proof of
interchangeable same-bus generators, with independent original-model replay.
It cannot excuse failed solves, infeasibility, changed objectives, or other
physical discrepancies. See [GPU execution for HVDC OPF](@ref) for the sharing
criteria and CPU-only replay tools.

The output contains frozen cases, Project/Manifest files, primal records,
derivative reports, strict and sharing comparisons, individual outcomes, and a
summary. Exceptions return a nonzero process status and may leave partial
evidence; use a new output directory for every attempt.

## Status compatibility and tests

The environment retains the validated CUDA 5/MadNLP 0.9 stack and permits
ExaModels `0.9.7` with MadNLP `0.9.2`. ExaModels'
adapter in that release expects symbolic solver statuses, while MadNLP returns
its own enum. The example supplies an owned callable solver adapter and extends
ExaModels' translation hooks only for that adapter type. It preserves native
MadNLP statuses and does not patch installed dependency sources. Unknown/error
statuses cannot become a feasible successful result.

The CPU suite checks status translation for every MadNLP status, actual success,
iteration limit, infeasibility, invalid evaluations, isolated derivative errors,
and both small HVDC conversion cases. The explicit GPU suite adds GPU solves
and derivative comparisons:

```sh
julia --project=tutorials/examodels tutorials/examodels/runtests.jl
julia --project=tutorials/examodels tutorials/examodels/runtests.jl --gpu
```

To use the helper from Julia, load GPU packages before calling its functions:

```julia
using CUDA, CUDSS, MadNLPGPU
include("tutorials/examodels/opf.jl")
result, pm, solver = ExaHVDC.solve(; gpu_modules=(;CUDA,CUDSS,MadNLPGPU))
```

`ExaHVDC.solve()` without `gpu_modules` runs on the CPU. Use `run.jl` for the
complete acceptance workflow and persisted evidence.
