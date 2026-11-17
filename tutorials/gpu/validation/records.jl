# Included after sharing.jl. Record capture is separate so the comparator's
# CPU-only unit tests do not import MadNLP or any CUDA package.
module GPUValidationRecords

using JuMP, PowerModels, PowerModelsACDC, MadNLP, SHA, TOML
using ..VerifiedSharing

"Capture a solved JuMP model in the portable validation TOML format."
function capture(result, pm, casefile; gpu_modules=nothing)
    raw = JuMP.unsafe_backend(pm.model)
    solver = raw.solver
    gpu = gpu_modules !== nothing
    storage = Dict{String,Any}("model_evaluation_backend"=>"cpu",
        "nlp_vector_type"=>string(typeof(raw.nlp.meta.x0)),
        "solver_vector_type"=>string(typeof(MadNLP.full(solver.x))),
        "linear_solver_type"=>string(typeof(solver.kkt.linear_solver)))
    if gpu
        cuda = gpu_modules.CUDA
        storage["gpu_nlp_vector"] = raw.nlp.meta.x0 isa cuda.CuArray
        storage["gpu_solver_vector"] = MadNLP.full(solver.x) isa cuda.CuArray
        storage["gpu_factorization_matrix"] = solver.kkt.linear_solver.tril isa cuda.CUSPARSE.CuSparseMatrixCSC
        storage["gpu_linear_solver"] = solver.kkt.linear_solver isa gpu_modules.MadNLPGPU.CUDSSSolver
        storage["cpu_evaluator"] = raw.nlp.inner.meta.x0 isa Vector{Float64}
        storage["cuDSS_info"] = Int(gpu_modules.CUDSS.cudss_get(solver.kkt.linear_solver.inner,"info"))
    end
    execution_checks = gpu ? Dict("gpu_storage"=>all(storage[k] for k in
        ("gpu_nlp_vector","gpu_solver_vector","gpu_factorization_matrix","gpu_linear_solver","cpu_evaluator")),
        "cuDSS_success"=>storage["cuDSS_info"] == 0) : Dict{String,Bool}()
    device = gpu ? Dict("name"=>gpu_modules.CUDA.name(gpu_modules.CUDA.device()),
        "uuid"=>string(gpu_modules.CUDA.uuid(gpu_modules.CUDA.device()))) : nothing
    return capture_record(result,pm,casefile,solver; native_status=raw.result.status,
        storage,execution_checks,gpu,device)
end

"Shared physical validation; backend adapters must supply their execution checks."
function capture_record(result,pm,casefile,solver; native_status,storage,execution_checks,
    gpu=false,device=nothing,versions=Dict{String,String}())
    options = MadNLP.get_opt(solver)
    record = merge(VerifiedSharing.fingerprints(pm.model), Dict{String,Any}(
        "format_version"=>1,"mode"=>gpu ? "gpu" : "cpu",
        "case"=>basename(casefile),"case_sha256"=>bytes2hex(sha256(read(casefile))),
        "formulation"=>"acp","variant"=>"base",
        "termination_status"=>string(result["termination_status"]),
        "primal_status"=>string(result["primal_status"]),
        "native_status"=>string(native_status),
        "objective"=>result["objective"],"solution"=>result["solution"],
        "variable_values"=>Dict(JuMP.name(v)=>JuMP.value(v) for v in JuMP.all_variables(pm.model)),
        "storage"=>storage,"execution_checks"=>execution_checks,
        "input_component_counts"=>Dict(k=>length(get(pm.data,k,Dict())) for k in VerifiedSharing.TABLES),
        "versions"=>Dict("Julia"=>string(VERSION),"JuMP"=>string(pkgversion(JuMP)),
            "PowerModels"=>string(pkgversion(PowerModels)),"PowerModelsACDC"=>string(pkgversion(PowerModelsACDC)),
            "MadNLP"=>string(pkgversion(MadNLP)),versions...),
        "resolved_options"=>Dict("tol"=>options.tol,"acceptable_tol"=>options.acceptable_tol,
            "bound_relax_factor"=>options.bound_relax_factor,
            "equality_treatment"=>string(options.equality_treatment),
            "fixed_variable_treatment"=>string(options.fixed_variable_treatment))))
    generators = VerifiedSharing.generator_map(pm)
    templates = Dict(k=>get(PowerModels.sol(pm),Symbol(k),Dict()) for k in VerifiedSharing.TABLES)
    checks, detail = VerifiedSharing.replay(pm.model,generators,record;templates=templates)
    merge!(record,detail)
    record["passed"] = all(values(checks)) &&
        result["termination_status"] in (JuMP.MOI.LOCALLY_SOLVED,JuMP.MOI.OPTIMAL) &&
        result["primal_status"] == JuMP.MOI.FEASIBLE_POINT && native_status == MadNLP.SOLVE_SUCCEEDED &&
        all(!isempty(result["solution"][k]) for k in ("busdc","branchdc","convdc")) &&
        all(values(execution_checks))
    record["original_checks"] = checks
    device === nothing || (record["device"] = device)
    return record
end

function save_record(path,record)
    ispath(path) && error("Refusing to overwrite $path")
    open(path,"w") do io
        TOML.print(io,record;sorted=true)
    end
end

"An explicitly selected sharing result cannot mask failed individual solves."
function accepted(cpu,gpu,strict,sharing,mode)
    mode in ("strict","verified-sharing") || error("Unknown comparison mode $mode")
    selected = mode == "strict" ? strict : sharing
    return cpu["passed"] && gpu["passed"] && selected["passed"]
end

end
