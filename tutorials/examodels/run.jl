# Reuse the existing case selection, CPU reference, record and comparison tools.
include("../gpu/validation/run.jl")
include("opf.jl")
include("derivatives.jl")

function run_exa_validation(args; cpu_only=false)
    options = validation_options(args)
    modules = nothing
    if !cpu_only
        all(isdefined(@__MODULE__,name) for name in (:CUDA,:CUDSS,:MadNLPGPU)) ||
            error("Load CUDA, CUDSS, MadNLPGPU before running GPU validation")
        CUDA.functional() || error("CUDA is not functional; no CPU fallback is permitted")
        CUDA.allowscalar(false)
        modules = (;CUDA,CUDSS,MadNLPGPU)
    end
    mkpath(options.output)
    project = Base.active_project()
    cp(project,joinpath(options.output,"Project.toml"))
    manifest = joinpath(dirname(project),"Manifest.toml")
    isfile(manifest) && cp(manifest,joinpath(options.output,"Manifest.toml"))
    entries = Dict{String,Any}[]
    modes = cpu_only ? ("jump_cpu","exa_cpu") : ("jump_cpu","exa_cpu","exa_gpu")
    for (index,casefile) in enumerate(options.paths)
        folder = joinpath(options.output,"$(index)_$(splitext(basename(casefile))[1])")
        mkdir(folder)
        frozen_case = joinpath(folder,"case.m")
        cp(casefile,frozen_case)
        records = Dict{String,Any}()
        for mode in modes
            if mode == "jump_cpu"
                result,pm = solve_hvdc_example(;gpu=false,casefile=frozen_case,
                    matched_treatments=true,max_wall_time=300.0)
                record = GPUValidationRecords.capture(result,pm,frozen_case)
            else
                gpu_modules = mode == "exa_gpu" ? modules : nothing
                result,pm,solver = ExaHVDC.solve(;casefile=frozen_case,gpu_modules)
                record = ExaHVDC.capture(result,pm,frozen_case,solver;gpu_modules)
                derivative_report = ExaDerivativeChecks.validate(pm;gpu_modules)
                GPUValidationRecords.save_record(joinpath(folder,mode*"_derivatives.toml"),derivative_report)
                record["execution_checks"]["original_expression_derivatives"] = derivative_report["passed"]
                record["passed"] &= derivative_report["passed"]
            end
            GPUValidationRecords.save_record(joinpath(folder,mode*".toml"),record)
            records[mode] = record
        end
        comparisons = Dict{String,Any}()
        pairs = cpu_only ? [("jump_cpu","exa_cpu")] :
            [("jump_cpu","exa_cpu"),("exa_cpu","exa_gpu"),("jump_cpu","exa_gpu")]
        for (a,b) in pairs
            label = a*"_vs_"*b
            strict = VerifiedSharing.compare_records(records[a],records[b])
            GPUValidationRecords.save_record(joinpath(folder,label*"_strict.toml"),strict)
            sharing_path = joinpath(folder,label*"_sharing.toml")
            VerifiedSharing.compare_sharing_files([joinpath(folder,a*".toml"),joinpath(folder,b*".toml"),frozen_case,sharing_path])
            sharing = TOML.parsefile(sharing_path)
            comparisons[label] = Dict("strict_passed"=>strict["passed"],"sharing_passed"=>sharing["passed"],
                "objective_relative_difference"=>strict["objective_relative_difference"],
                "passed"=>GPUValidationRecords.accepted(records[a],records[b],strict,sharing,options.comparison))
        end
        outcome = Dict("case"=>basename(casefile),"directory"=>basename(folder),
            "comparison_mode"=>options.comparison,"comparisons"=>comparisons,
            "individual_passed"=>Dict(k=>v["passed"] for (k,v) in records),
            "passed"=>all(v["passed"] for v in values(comparisons)))
        GPUValidationRecords.save_record(joinpath(folder,"outcome.toml"),outcome)
        push!(entries,outcome)
        println("EXAMODELS ",basename(casefile)," passed=",outcome["passed"])
        GC.gc()
    end
    summary = Dict("passed"=>all(e["passed"] for e in entries),
        "gpu_validated"=>!cpu_only,"modes"=>collect(modes),"runs"=>entries)
    GPUValidationRecords.save_record(joinpath(options.output,"summary.toml"),summary)
    return summary["passed"]
end

if abspath(PROGRAM_FILE) == @__FILE__
    cpu_only = "--cpu-only" in ARGS
    cpu_only || (@eval using CUDA, CUDSS, MadNLPGPU)
    run_exa_validation(filter(!=("--cpu-only"),ARGS);cpu_only) || exit(1)
end
