using Test, JuMP, TOML
include("sharing.jl")
const VS = VerifiedSharing

function fixture(; quadratic=false, reactive_cost=false, coupled=false, nonlinear=false)
    model = Model()
    @variable(model, 0 <= pg[1:4] <= 10, start=2)
    @variable(model, -10 <= qg[1:4] <= 10, start=1)
    @variable(model, -1 <= flow <= 1, start=0)
    @constraint(model, sum(pg[1:3]) == 6)
    @constraint(model, pg[4] == 2)
    @constraint(model, sum(qg[1:3]) == 3)
    @constraint(model, qg[4] == 1)
    @constraint(model, flow == 0)
    @objective(model,Min,pg[1]+pg[2]+2pg[3]+pg[4] +
        (quadratic ? pg[1]^2+pg[2]^2 : 0) + (reactive_cost ? qg[1]+qg[2]+qg[3] : 0))
    coupled && @constraint(model,qg[1]+qg[2] <= 5)
    nonlinear && @constraint(model,sin(qg[1])+sin(qg[2]) <= 2)
    generators = Dict(string(i)=>(bus=i == 4 ? "2" : "1",pg=pg[i],qg=qg[i]) for i in 1:4)
    point = Dict(name(v)=>(v == flow ? 0.0 : startswith(name(v),"pg") ? 2.0 : 1.0) for v in all_variables(model))
    function record(p)
        sol = Dict{String,Any}(table=>Dict{String,Any}() for table in VS.TABLES)
        sol["gen"] = Dict(id=>Dict("pg"=>p[name(g.pg)],"qg"=>p[name(g.qg)]) for (id,g) in generators)
        sol["bus"] = Dict("1"=>Dict("vm"=>1.0,"va"=>0.0),"2"=>Dict("vm"=>1.0,"va"=>0.0))
        sol["branch"] = Dict("1"=>Dict("pf"=>p["flow"]))
        return merge(VS.fingerprints(model),Dict("case_sha256"=>"fixture",
            "formulation"=>"acp","variant"=>"base","passed"=>true,
            "termination_status"=>"LOCALLY_SOLVED","primal_status"=>"FEASIBLE_POINT",
            "native_status"=>"SOLVE_SUCCEEDED","objective"=>value(v->p[name(v)],objective_function(model)),
            "solution"=>sol,"variable_values"=>copy(p)))
    end
    return model,generators,point,record
end

@testset "Verified generator sharing" begin
    m,g,p,record = fixture()
    cpu = record(p)
    q = copy(p); q["pg[1]"]+=0.01; q["pg[2]"]-=0.01
    q["qg[1]"]+=0.02; q["qg[3]"]-=0.02
    gpu = record(q)
    report = VS.compare_sharing(cpu,gpu,m,g)
    @test report["passed"]
    @test !report["strict_passed"]
    @test length(report["groups"]) == 2
    @test any(x->x["field"]=="pg" && x["generators"]==["1","2"],report["groups"])
    @test all(v->v<=1e-12,[x["abs_difference"] for x in report["groups"]])
    @test VS.compare_sharing(cpu,cpu,m,g)["passed"]
    @test VS.compare_sharing(cpu,cpu,m,g)["strict_passed"]

    @testset "Out of bounds despite identical group totals and stale success flags" begin
        bad = copy(p);bad["pg[1]"]=10.1;bad["pg[2]"]=-6.1
        r = VS.compare_sharing(cpu,record(bad),m,g)
        @test !r["passed"]
        @test r["checks"]["verified_group_totals_agree"]
        @test !r["checks"]["gpu_original_bounds"]
    end
    @testset "Movement across buses" begin
        bad = copy(p);bad["qg[1]"]+=0.01;bad["qg[4]"]-=0.01
        r = VS.compare_sharing(cpu,record(bad),m,g)
        @test !r["passed"]
        @test !r["checks"]["gpu_original_constraints"]
        @test !r["checks"]["verified_group_totals_agree"]
    end
    @testset "Different marginal costs" begin
        bad = copy(p);bad["pg[1]"]+=0.01;bad["pg[3]"]-=0.01
        r = VS.compare_sharing(cpu,record(bad),m,g)
        @test !r["passed"]
        @test !r["checks"]["individual_fields_agree"]
        @test !r["checks"]["objective_agreement"]
    end
    @testset "Group tolerance is not multiplied by generator count" begin
        bad = copy(p)
        for i in 1:3;bad["qg[$i]"]+=0.6e-4;end
        r = VS.compare_sharing(cpu,record(bad),m,g)
        @test !r["checks"]["verified_group_totals_agree"]
        @test !r["passed"]
    end
    @testset "Reported fields remain bound to checked primal point" begin
        bad = deepcopy(gpu);bad["solution"]["gen"]["1"]["qg"]+=0.01
        bad["solution"]["gen"]["2"]["qg"]-=0.01
        r=VS.compare_sharing(cpu,bad,m,g)
        @test !r["passed"]
        @test !r["checks"]["gpu_reported_generation_matches_primal"]
    end
    @testset "Network field discrepancies" begin
        for table in ("bus","branch","busdc","branchdc","convdc")
            a,b=deepcopy(cpu),deepcopy(gpu)
            a["solution"][table]=Dict("x"=>Dict("physical"=>1.0))
            b["solution"][table]=Dict("x"=>Dict("physical"=>1.001))
            r=VS.compare_sharing(a,b,m,g)
            @test !r["passed"]
            @test !r["checks"]["individual_fields_agree"]
        end
    end
    @testset "All physical output tables are bound to the primal point" begin
        templates = Dict{String,Any}(t=>Dict() for t in VS.TABLES)
        templates["bus"] = cpu["solution"]["bus"]
        templates["branch"] = Dict("1"=>Dict("pf"=>variable_by_name(m,"flow")))
        templates["gen"] = Dict(id=>Dict("pg"=>v.pg,"qg"=>v.qg) for (id,v) in g)
        @test VS.compare_sharing(cpu,gpu,m,g;templates=templates)["passed"]
        a,b=deepcopy(cpu),deepcopy(gpu)
        # Both records agreeing on a false output must still be rejected.
        for r in (a,b);r["solution"]["bus"]["1"]["vm"]=1.01;end
        r=VS.compare_sharing(a,b,m,g;templates=templates)
        @test !r["passed"]
        @test !r["checks"]["gpu_all_reported_physical_fields_match_primal"]
    end
    @testset "Provenance, finiteness, statuses and schema" begin
        bad=deepcopy(gpu);bad["objective"]+=1
        @test !VS.compare_sharing(cpu,bad,m,g)["checks"]["gpu_recorded_objective_matches_model"]
        bad=deepcopy(gpu);bad["variable_values"]["qg[1]"]=NaN
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
        bad=deepcopy(gpu);delete!(bad["variable_values"],"flow")
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
        bad=deepcopy(gpu);bad["native_status"]="MAXIMUM_ITERATIONS_EXCEEDED"
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
        bad=deepcopy(gpu);bad["case_sha256"]="different"
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
        @test !VS.compare_sharing(cpu,gpu,m,g;expected=Dict("model_sha256"=>"wrong"))["passed"]
        bad=deepcopy(gpu);bad["variant"]="zero_dc_r"
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
        # Added output components must not be silently ignored.
        bad=deepcopy(gpu);bad["solution"]["gen"]["5"]=Dict("pg"=>0.0,"qg"=>0.0)
        @test !VS.compare_sharing(cpu,bad,m,g)["passed"]
    end
    @testset "Fail closed on structurally ineligible generators" begin
        for kwargs in ((quadratic=true,),(reactive_cost=true,),(coupled=true,),(nonlinear=true,))
            other,gens,vals,make=fixture(;kwargs...)
            groups,excluded=VS.prove_groups(other,gens)
            field=haskey(kwargs,:quadratic) ? "pg" : "qg"
            @test all(x->!(x["field"]==field && "1" in x["generators"]),groups)
            @test haskey(excluded,"1.$field")
        end
        other,gens,vals,make=fixture()
        fix(gens["1"].pg,2.0;force=true)
        groups,excluded=VS.prove_groups(other,gens)
        @test haskey(excluded,"1.pg")
        # Same aggregate quadratic cost shape does not make Pg interchangeable.
        other,gens,vals,make=fixture(quadratic=true)
        changed=copy(vals);changed["pg[1]"]+=0.01;changed["pg[2]"]-=0.01
        @test !VS.compare_sharing(make(vals),make(changed),other,gens)["passed"]
        other,gens,vals,make=fixture()
        set_integer(gens["1"].pg)
        @test !VS.compare_sharing(make(vals),make(vals),other,gens)["passed"]
    end
    @testset "TOML serialization and no mutation" begin
        old_cpu,old_gpu=deepcopy(cpu),deepcopy(gpu)
        mktemp() do path,io
            TOML.print(io,report;sorted=true);close(io)
            @test TOML.parsefile(path)["passed"]
        end
        @test cpu==old_cpu && gpu==old_gpu
    end
    @testset "Deterministic diagnostic component for tied differences" begin
        identical = VS.compare_records(cpu,cpu)
        @test identical["physical_differences"]["gen.pg"]["component"] == "1"
        @test identical["physical_differences"]["gen.qg"]["component"] == "1"
        mktemp() do path,io
            TOML.print(io,cpu;sorted=true);close(io)
            saved=TOML.parsefile(path)
            @test VS.compare_records(saved,saved) == identical
        end
    end
end
