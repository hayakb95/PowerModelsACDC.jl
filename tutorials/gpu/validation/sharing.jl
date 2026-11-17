# Match opf.jl's Main import when displaying the historical model fingerprint:
# Julia otherwise qualifies JuMP type names in ScalarConstraint's default show.
using JuMP

module VerifiedSharing

using JuMP, PowerModels, PowerModelsACDC, SHA, TOML
include("compare.jl")

const FEASIBILITY_ATOL = 1e-6
const PHYSICAL_ATOL = 1e-4
const OBJECTIVE_RTOL = 1e-6
const TABLES = ("bus", "gen", "branch", "busdc", "branchdc", "convdc")
sha_bytes(x) = bytes2hex(sha256(Vector{UInt8}(codeunits(x))))

"Rebuild the same unsolved ACP/base model used by opf.jl; no optimizer is loaded."
function rebuild(casefile)
    data = PowerModelsACDC.parse_file(abspath(casefile))
    refs = [PowerModelsACDC.add_ref_dcgrid!, PowerModelsACDC.ref_add_pst!,
        PowerModelsACDC.ref_add_sssc!, PowerModelsACDC.ref_add_flex_load!,
        PowerModelsACDC.ref_add_gendc!, PowerModelsACDC.ref_add_im!]
    return PowerModels.instantiate_model(data, PowerModels.ACPPowerModel,
        PowerModelsACDC.build_acdcopf; ref_extensions=refs,
        setting=Dict("conv_losses_mp" => true))
end

function fingerprints(model; canonical_output=nothing)
    vars = JuMP.all_variables(model)
    names = JuMP.name.(vars)
    length(unique(names)) == length(names) || error("Nonunique variable names")
    starts = Float64[]
    for v in vars
        lo = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf
        hi = JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf
        push!(starts, clamp(something(JuMP.start_value(v), 0.0), lo, hi))
    end
    lines = ["variable: $(JuMP.name(v))" for v in vars]
    for (F,S) in JuMP.list_of_constraint_types(model), c in JuMP.all_constraints(model,F,S)
        push!(lines, "constraint: $(JuMP.constraint_object(c))")
    end
    push!(lines, "objective: $(JuMP.objective_sense(model)) $(JuMP.objective_function(model))")
    canonical = join(sort(lines), "\n")
    canonical_output === nothing || write(canonical_output,canonical)
    return Dict("model_sha256" => sha_bytes(canonical),
        "start_sha256" => sha_bytes(join(sort(["$(names[i])=$(repr(starts[i]))" for i in eachindex(starts)]), "\n")),
        "variable_count" => length(vars), "constraint_count_including_bounds" =>
            JuMP.num_constraints(model; count_variable_in_set_constraints=true))
end

# Conservative structural analysis: target variables in nonlinear expressions,
# quadratic terms, vector constraints, or unknown expressions cannot be grouped.
# No sampling, numerical derivatives, or coefficient tolerances are used.
function coefficients!(linear, unsafe, x, targets)
    if x isa Real
        isfinite(x) || error("Nonfinite model coefficient")
    elseif x isa JuMP.VariableRef
        x in targets && (linear[x] = get(linear, x, 0.0) + 1.0)
    elseif x isa JuMP.GenericAffExpr
        for (a,v) in JuMP.linear_terms(x)
            isfinite(a) || error("Nonfinite model coefficient")
            v in targets && a != 0 && (linear[v] = get(linear, v, 0.0) + a)
        end
    elseif x isa JuMP.GenericQuadExpr
        coefficients!(linear, unsafe, x.aff, targets)
        for (a,v,w) in JuMP.quad_terms(x)
            isfinite(a) || error("Nonfinite model coefficient")
            if a != 0
                v in targets && push!(unsafe, v)
                w in targets && push!(unsafe, w)
            end
        end
    elseif x isa JuMP.GenericNonlinearExpr || x isa AbstractArray
        args = x isa AbstractArray ? x : x.args
        for a in args
            found = Dict{JuMP.VariableRef,Float64}()
            coefficients!(found, unsafe, a, targets)
            union!(unsafe, keys(found))
        end
    else
        error("Unsupported expression type $(typeof(x)); sharing proof refused")
    end
    return nothing
end

"Prove that a redistribution preserves every non-bound constraint and the objective."
function prove_groups(model, generators)
    targets = Set(v for g in values(generators) for v in (g.pg, g.qg))
    signature = Dict(v => Tuple{Int,Float64}[] for v in targets)
    unsafe = Set{JuMP.VariableRef}()
    rows = Dict{Int,Any}()
    row = 0
    JuMP.num_nonlinear_constraints(model) == 0 || error("Legacy nonlinear constraints are unsupported")
    for (F,S) in JuMP.list_of_constraint_types(model), c in JuMP.all_constraints(model,F,S)
        obj = JuMP.constraint_object(c)
        # These bounds are checked separately on each full primal point.
        obj.func isa JuMP.VariableRef && obj.set isa Union{JuMP.MOI.EqualTo,JuMP.MOI.LessThan,JuMP.MOI.GreaterThan,JuMP.MOI.Interval} && continue
        row += 1
        a = Dict{JuMP.VariableRef,Float64}()
        coefficients!(a, unsafe, obj.func, targets)
        if !(obj.set isa JuMP.MOI.EqualTo)
            union!(unsafe, keys(a))
        end
        for (v,k) in a
            k != 0 && push!(signature[v], (row,k))
        end
        !isempty(a) && (rows[row] = c)
    end
    objective = Dict{JuMP.VariableRef,Float64}()
    coefficients!(objective, unsafe, JuMP.objective_function(model), targets)
    candidates = Dict{Any,Vector{String}}()
    excluded = Dict{String,Any}()
    for id in sort(collect(keys(generators))), field in ("pg", "qg")
        g = generators[id]
        v = field == "pg" ? g.pg : g.qg
        s = signature[v]
        cost = get(objective, v, 0.0)
        reason = v in unsafe ? "nonlinear, quadratic, vector or non-equality dependence" :
            JuMP.is_fixed(v) ? "fixed generator variable" :
            length(s) != 1 ? "requires exactly one non-bound equality" :
            abs(only(s)[2]) != 1 ? "requires unit balance coefficient" :
            field == "qg" && cost != 0 ? "reactive generation has an objective coefficient" : ""
        if !isempty(reason)
            excluded["$id.$field"] = reason
            continue
        end
        key = (g.bus, field, only(s), cost)
        push!(get!(candidates, key, String[]), id)
    end
    groups = Dict{String,Any}[]
    for (key, ids) in candidates
        length(ids) >= 2 || continue
        bus, field, entry, cost = key
        push!(groups, Dict("bus" => bus, "field" => field, "generators" => sort(ids),
            "constraint_row" => entry[1], "constraint" => string(rows[entry[1]]),
            "balance_coefficient" => entry[2], "objective_coefficient" => cost,
            "proof" => "same bus; identical single equality columns; no nonlinear/quadratic dependence; identical linear objective coefficients; individual bounds retained"))
    end
    sort!(groups; by=g -> (g["bus"],g["field"],join(g["generators"],",")))
    return groups, excluded
end

function generator_map(pm)
    return Dict(string(i) => (bus=string(g["gen_bus"]),
        pg=PowerModels.var(pm,:pg,i), qg=PowerModels.var(pm,:qg,i))
        for (i,g) in PowerModels.ref(pm,:gen))
end

function evaluate_template(x, point)
    x isa AbstractDict && return Dict(string(k)=>evaluate_template(v,point) for (k,v) in x)
    x isa Real && return x
    x isa Union{JuMP.VariableRef,JuMP.GenericAffExpr,JuMP.GenericQuadExpr,JuMP.GenericNonlinearExpr} && return JuMP.value(point,x)
    error("Unsupported solution template $(typeof(x)); replay refused")
end

function replay(model, generators, record; templates=nothing)
    vars = JuMP.all_variables(model)
    saved = record["variable_values"]
    schema = Set(JuMP.name.(vars)) == Set(keys(saved))
    ok = schema && finite_numeric(saved)
    checks = Dict{String,Bool}("same_variable_schema" => schema, "finite_primal_point" => ok)
    detail = Dict{String,Any}()
    if !ok
        return checks, detail
    end
    point(v) = Float64(saved[JuMP.name(v)])
    # Explicit finiteness check: NaN distances alone could escape > atol checks.
    checks["finite_constraint_evaluations"] = all(
        finite_numeric(JuMP.value.(point, JuMP.constraint_object(c).func))
        for (F,S) in JuMP.list_of_constraint_types(model) for c in JuMP.all_constraints(model,F,S))
    violations = JuMP.primal_feasibility_report(point,model; atol=0.0)
    bounds = 0.0
    constraints = 0.0
    for (c,value) in violations
        if JuMP.constraint_object(c).func isa JuMP.VariableRef
            bounds = max(bounds,value)
        else
            constraints = max(constraints,value)
        end
    end
    checks["original_bounds"] = isfinite(bounds) && bounds <= FEASIBILITY_ATOL
    checks["original_constraints"] = isfinite(constraints) && constraints <= FEASIBILITY_ATOL
    objective = JuMP.value(point,JuMP.objective_function(model))
    checks["recorded_objective_matches_model"] = isfinite(objective) && isfinite(record["objective"]) &&
        isapprox(objective,record["objective"]; rtol=1e-10,atol=1e-8)
    checks["same_active_generators"] = Set(keys(generators)) == Set(keys(record["solution"]["gen"]))
    # Bind reported generator outputs to the primal point whose bounds/equations
    # were just checked; a forged passed flag or stale residual cannot suffice.
    checks["reported_generation_matches_primal"] = checks["same_active_generators"] &&
        all(record["solution"]["gen"][id][field] == point(field == "pg" ? g.pg : g.qg)
            for (id,g) in generators for field in ("pg","qg"))
    if templates !== nothing
        rebuilt = evaluate_template(templates,point)
        checks["all_reported_physical_fields_match_primal"] = all(
            get(record["solution"],table,nothing) == rebuilt[table] for table in TABLES)
    end
    detail["max_original_bound_violation"] = bounds
    detail["max_original_constraint_violation"] = constraints
    detail["recomputed_objective"] = objective
    return checks, detail
end

function compare_sharing(cpu, gpu, model, generators; expected=Dict{String,Any}(), templates=nothing)
    strict = compare_records(cpu,gpu)
    checks = copy(strict["checks"])
    delete!(checks,"physical_solution_agreement")
    for (key,value) in expected, (mode,r) in (("cpu",cpu),("gpu",gpu))
        checks[mode * "_replay_" * key] = get(r,key,nothing) == value
    end
    checks["supported_scope"] = all(r["formulation"] == "acp" && r["variant"] == "base" for r in (cpu,gpu))
    checks["continuous_model"] = all(!JuMP.is_binary(v) && !JuMP.is_integer(v) for v in JuMP.all_variables(model))
    details = Dict{String,Any}()
    for (mode,r) in (("cpu",cpu),("gpu",gpu))
        checks[mode * "_successful_status"] = r["termination_status"] in ("LOCALLY_SOLVED","OPTIMAL") &&
            r["primal_status"] == "FEASIBLE_POINT" && r["native_status"] == "SOLVE_SUCCEEDED"
        rc, detail = replay(model,generators,r; templates=templates)
        for (k,v) in rc
            checks[mode * "_" * k] = v
        end
        details[mode] = detail
    end
    groups,excluded = prove_groups(model,generators)
    covered = Set((id,g["field"]) for g in groups for id in g["generators"])
    rows = Dict{String,Any}[]
    for table in TABLES
        a,b = cpu["solution"][table],gpu["solution"][table]
        for id in sort(collect(intersect(keys(a),keys(b)))), field in sort(collect(intersect(keys(a[id]),keys(b[id]))))
            x,y = a[id][field],b[id][field]
            if x isa Number && y isa Number
                shared = table == "gen" && (id,field) in covered
                push!(rows,Dict("table"=>table,"component"=>id,"field"=>field,
                    "cpu"=>x,"gpu"=>y,"abs_difference"=>abs(x-y),
                    "used_for_comparison"=>!shared,
                    "reason"=>shared ? "verified group total is checked separately" : "individual field"))
            else
                checks["same_nonnumeric_fields"] = get(checks,"same_nonnumeric_fields",true) && isequal(x,y)
            end
        end
    end
    for g in groups
        a = sum(cpu["solution"]["gen"][id][g["field"]] for id in g["generators"])
        b = sum(gpu["solution"]["gen"][id][g["field"]] for id in g["generators"])
        g["cpu_total"],g["gpu_total"],g["abs_difference"] = a,b,abs(a-b)
    end
    checks["individual_fields_agree"] = all(isfinite(r["abs_difference"]) && r["abs_difference"] <= PHYSICAL_ATOL for r in rows if r["used_for_comparison"])
    checks["verified_group_totals_agree"] = all(isfinite(g["abs_difference"]) && g["abs_difference"] <= PHYSICAL_ATOL for g in groups)
    return Dict("passed"=>all(values(checks)),"mode"=>"verified_generator_sharing",
        "strict_passed"=>strict["passed"],"strict"=>strict,"checks"=>checks,
        "thresholds"=>Dict("original_feasibility_atol"=>FEASIBILITY_ATOL,
            "physical_atol"=>PHYSICAL_ATOL,"group_total_atol"=>PHYSICAL_ATOL,
            "objective_rtol"=>OBJECTIVE_RTOL,"objective_replay_rtol"=>1e-10,"objective_replay_atol"=>1e-8),
        "groups"=>groups,"excluded_generator_fields"=>excluded,"replay"=>details,
        "individual_differences"=>rows,
        "scope"=>"Single-period continuous ACP/base OPF; agreement allowing only structurally verified sharing; no global optimality claim")
end

function compare_sharing_files(args)
    length(args) == 4 || error("Usage: sharing.jl cpu.toml gpu.toml case.m NEW_OUTPUT.toml")
    cpu_path,gpu_path,casefile,output = abspath.(args)
    ispath(output) && error("Refusing to overwrite $output")
    cpu,gpu = TOML.parsefile(cpu_path),TOML.parsefile(gpu_path)
    all(r["formulation"] == "acp" && r["variant"] == "base" for r in (cpu,gpu)) || error("Sharing comparison currently supports ACP/base only")
    pm = rebuild(casefile)
    length(PowerModels.nw_ids(pm)) == 1 || error("Only single-network models are supported")
    println("Rebuilt unsolved model; fingerprinting and replaying saved primal points")
    flush(stdout)
    canonical_output = output * ".rebuilt_model.txt"
    ispath(canonical_output) && error("Refusing to overwrite $canonical_output")
    expected = fingerprints(pm.model; canonical_output=canonical_output)
    expected["case_sha256"] = bytes2hex(sha256(read(casefile)))
    # Match the model-building package versions, independent of solver backends.
    versions = Dict("JuMP"=>string(pkgversion(JuMP)),"PowerModels"=>string(pkgversion(PowerModels)),
        "PowerModelsACDC"=>string(pkgversion(PowerModelsACDC)))
    sol = PowerModels.sol(pm)
    templates = Dict(table=>get(sol,Symbol(table),Dict()) for table in TABLES)
    report = compare_sharing(cpu,gpu,pm.model,generator_map(pm); expected=expected,templates=templates)
    for (mode,r) in (("cpu",cpu),("gpu",gpu)), (key,value) in versions
        report["checks"][mode * "_version_" * key] = get(r["versions"],key,"") == value
    end
    report["passed"] = all(values(report["checks"]))
    report["provenance"] = Dict("cpu_path"=>cpu_path,"gpu_path"=>gpu_path,"case_path"=>casefile,
        "cpu_sha256"=>bytes2hex(sha256(read(cpu_path))),"gpu_sha256"=>bytes2hex(sha256(read(gpu_path))),
        "rebuilt_fingerprints"=>expected,"model_versions"=>versions)
    open(output,"w") do io
        TOML.print(io,report; sorted=true)
    end
    println("SHARING strict_passed=",report["strict_passed"]," sharing_passed=",report["passed"]," groups=",length(report["groups"]))
    for (k,v) in sort(collect(report["checks"]))
        v || println("FAILED: ",k)
    end
    return report["passed"]
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    VerifiedSharing.compare_sharing_files(ARGS) || exit(1)
end
