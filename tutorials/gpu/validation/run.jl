# Portable optional GPU experiment: launch Julia on an allocated GPU node.
# There are no scheduler commands, workspace paths, or external depot assumptions.
using TOML
include("../opf.jl")
include("sharing.jl")
include("records.jl")

function validation_options(args)
    output = nothing
    comparison = "strict"
    cases = String[]
    large = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--output","--comparison")
            i += 1
            i <= length(args) || error("Missing value for $arg")
            if arg == "--output"
                output === nothing || error("Repeated --output")
                output = abspath(args[i])
            else
                comparison = args[i]
            end
        elseif arg == "--large"
            large = true
        elseif startswith(arg,"--")
            error("Unknown option $arg")
        else
            push!(cases,arg)
        end
        i += 1
    end
    output === nothing && error("Usage: run.jl --output NEW_DIRECTORY [--comparison strict|verified-sharing] [--large | case.m ...]")
    comparison in ("strict","verified-sharing") || error("Invalid comparison mode")
    large && !isempty(cases) && error("Use --large or explicit cases")
    if large
        cases = ["case39_acdc.m","pglib_opf_case588_sdet_acdc.m","case3120sp_acdc.m"]
    elseif isempty(cases)
        cases = ["case5_acdc.m","case5_lcc.m"]
    end
    paths = [isfile(p) ? abspath(p) : pkgdir(PowerModelsACDC,"test","data",p) for p in cases]
    all(isfile,paths) || error("Case file does not exist")
    ispath(output) && error("Refusing to overwrite $output")
    return (;output,comparison,paths)
end

function run_validation(args)
    options = validation_options(args)
    # The explicit GPU entry point fails when CUDA is unavailable.
    all(isdefined(@__MODULE__,name) for name in (:CUDA,:CUDSS,:MadNLPGPU)) ||
        error("Load CUDA, CUDSS, MadNLPGPU before calling run_validation")
    CUDA.functional() || error("Run validation on a CUDA-capable GPU")
    CUDA.allowscalar(false)
    mkpath(options.output)
    # Preserve the actual optional environment alongside generated evidence.
    project = Base.active_project()
    cp(project,joinpath(options.output,"Project.toml"))
    manifest = joinpath(dirname(project),"Manifest.toml")
    isfile(manifest) && cp(manifest,joinpath(options.output,"Manifest.toml"))
    entries = Dict{String,Any}[]
    for (index,casefile) in enumerate(options.paths)
        folder = joinpath(options.output,"$(index)_$(splitext(basename(casefile))[1])")
        mkdir(folder)
        frozen_case = joinpath(folder,"case.m")
        cp(casefile,frozen_case)
        records = Dict{String,Any}()
        for mode in ("cpu","gpu")
            result,pm = solve_hvdc_example(;gpu=mode=="gpu",casefile=frozen_case,
                matched_treatments=true,max_wall_time=300.0)
            modules = mode == "gpu" ? (;CUDA,CUDSS,MadNLPGPU) : nothing
            record = GPUValidationRecords.capture(result,pm,frozen_case;gpu_modules=modules)
            GPUValidationRecords.save_record(joinpath(folder,mode*".toml"),record)
            records[mode] = record
        end
        strict = VerifiedSharing.compare_records(records["cpu"],records["gpu"])
        GPUValidationRecords.save_record(joinpath(folder,"comparison.toml"),strict)
        sharing_path = joinpath(folder,"sharing_comparison.toml")
        VerifiedSharing.compare_sharing_files([joinpath(folder,"cpu.toml"),joinpath(folder,"gpu.toml"),frozen_case,sharing_path])
        sharing = TOML.parsefile(sharing_path)
        passed = GPUValidationRecords.accepted(records["cpu"],records["gpu"],strict,sharing,options.comparison)
        push!(entries,Dict("case"=>basename(casefile),"directory"=>basename(folder),
            "comparison_mode"=>options.comparison,"cpu_passed"=>records["cpu"]["passed"],
            "gpu_passed"=>records["gpu"]["passed"],"strict_passed"=>strict["passed"],
            "sharing_passed"=>sharing["passed"],"passed"=>passed))
        GPUValidationRecords.save_record(joinpath(folder,"outcome.toml"),last(entries))
        println(basename(casefile),": strict=",strict["passed"]," sharing=",sharing["passed"]," selected_passed=",passed)
        GC.gc()
    end
    summary = Dict("passed"=>all(e["passed"] for e in entries),"runs"=>entries)
    GPUValidationRecords.save_record(joinpath(options.output,"summary.toml"),summary)
    return summary["passed"]
end

if abspath(PROGRAM_FILE) == @__FILE__
    @eval using CUDA, CUDSS, MadNLPGPU
    run_validation(ARGS) || exit(1)
end
