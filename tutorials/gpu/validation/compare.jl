using TOML
finite_numeric(x::Number) = isfinite(x)
finite_numeric(x::AbstractDict) = all(finite_numeric, values(x))
finite_numeric(x::AbstractArray) = all(finite_numeric, x)
finite_numeric(x) = true
function compare_records(cpu, gpu)
    checks = Dict{String,Bool}("cpu_success" => cpu["passed"], "gpu_success" => gpu["passed"])
    for k in ("case_sha256", "model_sha256", "start_sha256", "variable_count", "constraint_count_including_bounds", "formulation", "variant")
        checks["same_" * k] = cpu[k] == gpu[k]
    end
    gap = abs(cpu["objective"] - gpu["objective"]) / max(1.0, abs(cpu["objective"]), abs(gpu["objective"]))
    checks["objective_agreement"] = gap <= 1e-6
    checks["finite_solutions"] = finite_numeric(cpu["solution"]) && finite_numeric(gpu["solution"])
    differences = Dict{String,Any}()
    for table in ("bus", "gen", "branch", "busdc", "branchdc", "convdc")
        a, b = cpu["solution"][table], gpu["solution"][table]
        checks["same_" * table * "_schema"] = keys(a) == keys(b) && all(Set(keys(a[i])) == Set(keys(b[i])) for i in intersect(keys(a),keys(b)))
        # Use a stable component order so equal maxima have the same diagnostic
        # owner in memory and after TOML serialization (including zero ties).
        for i in sort(collect(intersect(keys(a),keys(b)))), field in sort(collect(intersect(keys(a[i]),keys(b[i]))))
            x, y = a[i][field], b[i][field]
            if x isa Number && y isa Number
                key = table * "." * field
                gap_field = abs(x-y)
                if !haskey(differences, key) || gap_field > differences[key]["max_abs_difference"]
                    differences[key] = Dict("max_abs_difference" => gap_field, "component" => i)
                end
            end
        end
    end
    checks["physical_solution_agreement"] = all(v["max_abs_difference"] <= 1e-4 for v in values(differences))
    record = Dict("passed" => all(values(checks)), "checks" => checks, "objective_relative_difference" => gap,
        "cpu_objective" => cpu["objective"], "gpu_objective" => gpu["objective"], "physical_differences" => differences)
    return record
end
function compare_files(args)
    length(args) == 3 || error("Usage: compare.jl cpu.toml gpu.toml output.toml")
    ispath(args[3]) && error("Refusing to overwrite $(args[3])")
    cpu, gpu = TOML.parsefile.(args[1:2])
    record = compare_records(cpu, gpu)
    open(args[3], "w") do io
        TOML.print(io, record; sorted=true)
    end
    println("COMPARISON passed=", record["passed"], " relative_objective_gap=", record["objective_relative_difference"])
    return record["passed"]
end
if abspath(PROGRAM_FILE) == @__FILE__
    compare_files(ARGS) || exit(1)
end
