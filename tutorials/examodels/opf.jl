using JuMP, PowerModels, PowerModelsACDC
isdefined(@__MODULE__,:VerifiedSharing) || include("../gpu/validation/sharing.jl")
isdefined(@__MODULE__,:GPUValidationRecords) || include("../gpu/validation/records.jl")

module ExaHVDC
using JuMP, PowerModels, PowerModelsACDC, MadNLP, ExaModels, NLPModels
using ..VerifiedSharing, ..GPUValidationRecords
const MOI = JuMP.MOI

"An owned solver adapter: capture MadNLP and translate its native status enum."
struct MadNLPSolve
    solver::Base.RefValue{Any}
end
MadNLPSolve() = MadNLPSolve(Ref{Any}(nothing))
function (adapter::MadNLPSolve)(nlp; kwargs...)
    solver = MadNLP.MadNLPSolver(nlp; kwargs...)
    adapter.solver[] = solver
    return MadNLP.solve!(solver)
end

# Dispatch on our own adapter type; no dependency source patch or changes to
# ExaModels' methods for MadNLP.madnlp. Unknown statuses cannot become success.
function ExaModels.result_status_translator(::MadNLPSolve, status::MadNLP.Status)
    status == MadNLP.SOLVE_SUCCEEDED && return MOI.FEASIBLE_POINT
    status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL && return MOI.NEARLY_FEASIBLE_POINT
    status == MadNLP.INFEASIBLE_PROBLEM_DETECTED && return MOI.INFEASIBLE_POINT
    return MOI.UNKNOWN_RESULT_STATUS
end
function ExaModels.termination_status_translator(::MadNLPSolve, status::MadNLP.Status)
    return get(TERMINATION_CODES,status,MOI.OTHER_ERROR)
end
const TERMINATION_CODES = Dict(
    MadNLP.SOLVE_SUCCEEDED=>MOI.LOCALLY_SOLVED,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL=>MOI.ALMOST_LOCALLY_SOLVED,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL=>MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES=>MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED=>MOI.LOCALLY_INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED=>MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED=>MOI.TIME_LIMIT,
    MadNLP.INITIAL=>MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED=>MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED=>MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION=>MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM=>MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP=>MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR=>MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE=>MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT=>MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS=>MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN=>MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN=>MOI.INVALID_MODEL)

"Solve a single-network ACP HVDC model. Supply loaded CUDA modules for GPU mode."
function solve(; casefile=pkgdir(PowerModelsACDC,"test","data","case5_acdc.m"),
    gpu_modules=nothing,max_wall_time=300.0)
    gpu = gpu_modules !== nothing
    if gpu
        gpu_modules.CUDA.functional() || error("CUDA is not functional")
        gpu_modules.CUDA.allowscalar(false)
        Base.get_extension(gpu_modules.MadNLPGPU,:MadNLPGPUCUDAExt) === nothing &&
            error("The MadNLPGPU CUDA/cuDSS extension is not loaded")
        Base.get_extension(ExaModels,:ExaModelsCUDA) === nothing &&
            error("The ExaModels CUDA extension is not loaded")
        Base.get_extension(ExaModels,:ExaModelsKernelAbstractions) === nothing &&
            error("The ExaModels parallel evaluation extension is not loaded")
    end
    pm = VerifiedSharing.rebuild(casefile)
    for v in JuMP.all_variables(pm.model)
        lo = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
        hi = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
        JuMP.set_start_value(v,clamp(something(JuMP.start_value(v),0.0),lo,hi))
    end
    adapter = MadNLPSolve()
    backend = gpu ? gpu_modules.CUDA.CUDABackend() : nothing
    factory = () -> ExaModels.Optimizer(adapter,backend)
    optimizer = JuMP.optimizer_with_attributes(factory,
        "linear_solver"=>(gpu ? gpu_modules.MadNLPGPU.CUDSSSolver : MadNLP.MumpsSolver),
        "equality_treatment"=>MadNLP.RelaxEquality,
        "fixed_variable_treatment"=>MadNLP.RelaxBound,
        "tol"=>1e-8,"acceptable_tol"=>1e-8,"bound_relax_factor"=>1e-8,
        "max_iter"=>1000,"max_wall_time"=>max_wall_time,"blas_num_threads"=>1)
    result = PowerModels.optimize_model!(pm; optimizer)
    gpu && gpu_modules.CUDA.synchronize()
    return result,pm,adapter.solver[]
end

function capture(result,pm,casefile,solver; gpu_modules=nothing)
    gpu = gpu_modules !== nothing
    raw = JuMP.unsafe_backend(pm.model)
    nlp = raw.model
    storage = Dict{String,Any}("model_evaluation_backend"=>gpu ? "gpu" : "cpu",
        "modeler"=>"ExaModels","nlp_type"=>string(nameof(typeof(nlp))),
        "backend_type"=>string(typeof(raw.backend)),
        "nlp_vector_type"=>string(typeof(nlp.meta.x0)),
        "solver_vector_type"=>string(typeof(MadNLP.full(solver.x))),
        "linear_solver_type"=>string(typeof(solver.kkt.linear_solver)))
    checks = Dict{String,Bool}("exa_model"=>nlp isa ExaModels.ExaModel)
    device = nothing
    if gpu
        cuda = gpu_modules.CUDA
        storage["gpu_nlp_vector"] = nlp.meta.x0 isa cuda.CuArray
        storage["gpu_solver_vector"] = MadNLP.full(solver.x) isa cuda.CuArray
        storage["gpu_factorization_matrix"] = solver.kkt.linear_solver.tril isa cuda.CUSPARSE.CuSparseMatrixCSC
        storage["gpu_linear_solver"] = solver.kkt.linear_solver isa gpu_modules.MadNLPGPU.CUDSSSolver
        storage["cuDSS_info"] = Int(gpu_modules.CUDSS.cudss_get(solver.kkt.linear_solver.inner,"info"))
        checks["gpu_storage"] = all(storage[k] for k in
            ("gpu_nlp_vector","gpu_solver_vector","gpu_factorization_matrix","gpu_linear_solver"))
        checks["cuDSS_success"] = storage["cuDSS_info"] == 0
        checks["cuda_backend"] = raw.backend isa cuda.CUDABackend
        device = Dict("name"=>cuda.name(cuda.device()),"uuid"=>string(cuda.uuid(cuda.device())))
    else
        checks["cpu_storage"] = nlp.meta.x0 isa Vector{Float64} && MadNLP.full(solver.x) isa Vector{Float64}
    end
    return GPUValidationRecords.capture_record(result,pm,casefile,solver;
        native_status=raw.result.status,storage,execution_checks=checks,gpu,device,
        versions=Dict("ExaModels"=>string(pkgversion(ExaModels)),"NLPModels"=>string(pkgversion(NLPModels))))
end
end
