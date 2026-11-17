module ExaDerivativeChecks
using JuMP, ExaModels, NLPModels, SparseArrays, LinearAlgebra
const MOI = JuMP.MOI

# Per-entry normalized errors, rather than a global norm that could conceal a
# small incorrect equation alongside a large power-balance coefficient.
function agreement(a,b; atol=1e-7,rtol=1e-8)
    size(a) == size(b) || return Dict("passed"=>false,"same_shape"=>false)
    if issparse(a) && issparse(b)
        rows,cols,_ = findnz(abs.(a) + abs.(b))
        return agreement([a[i,j] for (i,j) in zip(rows,cols)],
            [b[i,j] for (i,j) in zip(rows,cols)];atol,rtol)
    end
    finite = all(isfinite,a) && all(isfinite,b)
    scaled = maximum(abs.(a .- b) ./ (atol .+ rtol .* max.(abs.(a),abs.(b)));init=0.0)
    return Dict("passed"=>finite && scaled <= 1.0,"same_shape"=>true,
        "finite"=>finite,"max_scaled_error"=>scaled,
        "max_absolute_error"=>maximum(abs.(a .- b);init=0.0),"atol"=>atol,"rtol"=>rtol)
end

"Independent MOI reverse AD, ordered by the actual JuMP-to-Exa optimizer map."
function reference(model)
    vars = sort(JuMP.all_variables(model);by=v->JuMP.optimizer_index(v).value)
    JuMP.optimizer_index.(vars) == MOI.VariableIndex.(1:length(vars)) || error("Unsupported variable map")
    constraints = [c for (F,S) in JuMP.list_of_constraint_types(model)
        for c in JuMP.all_constraints(model,F,S) if !(JuMP.constraint_object(c).func isa JuMP.VariableRef)]
    sort!(constraints;by=c->JuMP.optimizer_index(c).value)
    [JuMP.optimizer_index(c).value for c in constraints] == collect(1:length(constraints)) ||
        error("Unsupported constraint map")
    nl = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(nl,JuMP.moi_function(JuMP.objective_function(model)))
    for c in constraints
        obj = JuMP.constraint_object(c)
        MOI.Nonlinear.add_constraint(nl,JuMP.moi_function(obj.func),obj.set)
    end
    evaluator = MOI.Nonlinear.Evaluator(nl,MOI.Nonlinear.SparseReverseMode(),JuMP.index.(vars))
    MOI.initialize(evaluator,[:Grad,:Jac,:Hess])
    return evaluator,vars,constraints
end

function coordinate_matrix(pattern,values,m,n; symmetric=false)
    rows = first.(pattern)
    cols = last.(pattern)
    if symmetric
        rows,cols = max.(rows,cols),min.(rows,cols)
    end
    # ExaModels may emit repeated coordinates; sparse sums them before comparison.
    return sparse(rows,cols,values,m,n)
end

"Compare original expressions and derivatives at starts, a perturbed point, and solution."
function validate(pm; gpu_modules=nothing)
    raw = JuMP.unsafe_backend(pm.model)
    nlp = raw.model
    evaluator,vars,cons = reference(pm.model)
    n,m = length(vars),length(cons)
    n == nlp.meta.nvar && m == nlp.meta.ncon || error("Conversion dimensions differ")
    start = Float64[JuMP.start_value(v) for v in vars]
    lower,upper = Array(nlp.meta.lvar),Array(nlp.meta.uvar)
    perturbed = clamp.(start .+ 1e-3 .* sin.(collect(1:n)),lower,upper)
    points = [("start",start),("perturbed",perturbed),("solution",Array(raw.result.solution))]
    # Every multiplier is nonzero; objective-only and constraint-only curvature
    # are checked separately as well as the Hessian of their sum.
    multipliers = [0.2 + 0.01 * mod(i,17) for i in 1:m]
    rj = MOI.jacobian_structure(evaluator)
    rh = MOI.hessian_lagrangian_structure(evaluator)
    jrows,jcols = similar(nlp.meta.x0,Int,nlp.meta.nnzj),similar(nlp.meta.x0,Int,nlp.meta.nnzj)
    hrows,hcols = similar(nlp.meta.x0,Int,nlp.meta.nnzh),similar(nlp.meta.x0,Int,nlp.meta.nnzh)
    NLPModels.jac_structure!(nlp,jrows,jcols)
    NLPModels.hess_structure!(nlp,hrows,hcols)
    ej = collect(zip(Array(jrows),Array(jcols)))
    eh = collect(zip(Array(hrows),Array(hcols)))
    checks = Dict{String,Any}()
    # Infinite matching bounds are compared exactly, outside finite-only checks.
    expected_lower = [JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_lower_bound(v) ? JuMP.lower_bound(v) : -Inf for v in vars]
    expected_upper = [JuMP.is_fixed(v) ? JuMP.fix_value(v) : JuMP.has_upper_bound(v) ? JuMP.upper_bound(v) : Inf for v in vars]
    checks["variable_bounds"] = Dict("passed"=>lower == expected_lower && upper == expected_upper)
    bound(set::MOI.EqualTo) = (set.value,set.value)
    bound(set::MOI.LessThan) = (-Inf,set.upper)
    bound(set::MOI.GreaterThan) = (set.lower,Inf)
    bound(set::MOI.Interval) = (set.lower,set.upper)
    bounds = [bound(JuMP.constraint_object(c).set) for c in cons]
    checks["constraint_bounds"] = Dict("passed"=>Array(nlp.meta.lcon) == first.(bounds) && Array(nlp.meta.ucon) == last.(bounds))
    on_device = Dict{String,Bool}()
    to_backend(x) = gpu_modules === nothing ? copy(x) : gpu_modules.CUDA.CuArray(x)
    for (label,point) in points
        x = to_backend(point)
        c,g,j,h = similar(x,m),similar(x,n),similar(x,nlp.meta.nnzj),similar(x,nlp.meta.nnzh)
        objective = NLPModels.obj(nlp,x)
        NLPModels.cons!(nlp,x,c)
        NLPModels.grad!(nlp,x,g)
        NLPModels.jac_coord!(nlp,x,j)
        if gpu_modules !== nothing
            for (key,value) in (("input",x),("constraints",c),("gradient",g),("jacobian",j),("hessian",h))
                on_device[key] = get(on_device,key,true) && value isa gpu_modules.CUDA.CuArray
            end
            gpu_modules.CUDA.synchronize()
        end
        rc,rg,rjv,rhv = zeros(m),zeros(n),zeros(length(rj)),zeros(length(rh))
        MOI.eval_constraint(evaluator,rc,point)
        MOI.eval_objective_gradient(evaluator,rg,point)
        MOI.eval_constraint_jacobian(evaluator,rjv,point)
        checks[label*"_objective"] = agreement([objective],[MOI.eval_objective(evaluator,point)])
        checks[label*"_constraints"] = agreement(Array(c),rc)
        checks[label*"_gradient"] = agreement(Array(g),rg)
        checks[label*"_jacobian"] = agreement(coordinate_matrix(ej,Array(j),m,n),coordinate_matrix(rj,rjv,m,n))
        for (kind,weight,y) in (("objective",1.0,zeros(m)),("constraints",0.0,multipliers),("lagrangian",1.0,multipliers))
            yy = to_backend(y)
            NLPModels.hess_coord!(nlp,x,yy,h;obj_weight=weight)
            gpu_modules === nothing || gpu_modules.CUDA.synchronize()
            MOI.eval_hessian_lagrangian(evaluator,rhv,point,weight,y)
            checks[label*"_hessian_"*kind] = agreement(coordinate_matrix(eh,Array(h),n,n;symmetric=true),
                coordinate_matrix(rh,rhv,n,n;symmetric=true))
        end
    end
    gpu_modules === nothing || (checks["gpu_evaluation_arrays"] = Dict("passed"=>all(values(on_device))))
    return Dict("passed"=>all(c["passed"] for c in values(checks)),"checks"=>checks,
        "reference"=>"MOI sparse reverse AD on original JuMP expressions",
        "points"=>first.(points),"nonzero_constraint_multipliers"=>all(!iszero,multipliers),
        "gpu_arrays"=>on_device,"variable_count"=>n,"constraint_count"=>m)
end
end
