using JuMP, PowerModels, PowerModelsACDC, MadNLP

"""
    solve_hvdc_example(; gpu=true, casefile=nothing, formulation=ACPPowerModel,
                       matched_treatments=false, max_wall_time=Inf)

Solve the bundled meshed VSC case or an AC/DC case supplied by the caller.
With `gpu=true`, MadNLP's solver arrays and cuDSS factorization use CUDA;
JuMP's function and derivative evaluations remain on the CPU.
Returns the standard PowerModels result and the instantiated model for checks.
"""
function solve_hvdc_example(;
    gpu=true,
    casefile=pkgdir(PowerModelsACDC, "test", "data", "case5_acdc.m"),
    formulation=PowerModels.ACPPowerModel,
    matched_treatments=false,
    max_wall_time=Inf,
)
    # Load CUDA, CUDSS and MadNLPGPU before calling this function in GPU mode.
    if gpu
        all(isdefined(@__MODULE__, name) for name in (:CUDA, :CUDSS, :MadNLPGPU)) ||
            error("Load CUDA, CUDSS, MadNLPGPU first")
        CUDA.functional() || error("CUDA is not functional; use a GPU allocation")
        CUDA.allowscalar(false)
        Base.get_extension(MadNLPGPU, :MadNLPGPUCUDAExt) === nothing &&
            error("The MadNLPGPU CUDA/cuDSS extension is not loaded")
    end
    data = PowerModelsACDC.parse_file(casefile)
    refs = [PowerModelsACDC.add_ref_dcgrid!, PowerModelsACDC.ref_add_pst!,
        PowerModelsACDC.ref_add_sssc!, PowerModelsACDC.ref_add_flex_load!,
        PowerModelsACDC.ref_add_gendc!, PowerModelsACDC.ref_add_im!]
    pm = PowerModels.instantiate_model(data, formulation, PowerModelsACDC.build_acdcopf;
        ref_extensions=refs, setting=Dict("conv_losses_mp" => true))
    # Make the CPU and GPU inputs identical, including starts absent from data.
    for v in JuMP.all_variables(pm.model)
        lo = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
        hi = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
        JuMP.set_start_value(v, clamp(something(JuMP.start_value(v), 0.0), lo, hi))
    end
    options = Dict{String,Any}("tol" => 1e-8, "acceptable_tol" => 1e-8,
        "bound_relax_factor" => 1e-8, "max_iter" => 1000, "blas_num_threads" => 1)
    if matched_treatments
        # Use the condensed GPU system's treatments on both backends, while
        # validating the original JuMP constraints and bounds after the solve.
        options["equality_treatment"] = MadNLP.RelaxEquality
        options["fixed_variable_treatment"] = MadNLP.RelaxBound
    end
    isfinite(max_wall_time) && (options["max_wall_time"] = max_wall_time)
    if gpu
        options["array_type"] = CUDA.CuArray
        options["linear_solver"] = MadNLPGPU.CUDSSSolver
    else
        options["linear_solver"] = MadNLP.MumpsSolver
    end
    optimizer = JuMP.optimizer_with_attributes(MadNLP.Optimizer, collect(options)...)
    result = PowerModels.optimize_model!(pm; optimizer=optimizer)
    gpu && CUDA.synchronize()
    return result, pm
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = !("--cpu" in ARGS)
    if gpu
        @eval using CUDA, CUDSS, MadNLPGPU
    end
    paths = filter(arg -> arg != "--cpu", ARGS)
    length(paths) <= 1 || error("Usage: opf.jl [--cpu] [case.m]")
    casefile = isempty(paths) ? pkgdir(PowerModelsACDC, "test", "data", "case5_acdc.m") : only(paths)
    result, pm = solve_hvdc_example(; gpu, casefile)
    println("Status: ", result["termination_status"], "; objective: ", result["objective"])
    violations = JuMP.primal_feasibility_report(pm.model; atol=0.0)
    worst = maximum(values(violations); init=0.0)
    println("Maximum original-model violation: ", worst)
    result["termination_status"] in (JuMP.MOI.LOCALLY_SOLVED, JuMP.MOI.OPTIMAL) || error("Solve did not succeed")
    result["primal_status"] == JuMP.MOI.FEASIBLE_POINT || error("No feasible point")
    worst <= 1e-6 || error("Original-model feasibility check failed")
end
