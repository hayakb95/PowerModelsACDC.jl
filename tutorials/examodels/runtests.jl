using Test
include("run.jl")

function status_model(kind)
    adapter = ExaHVDC.MadNLPSolve()
    factory = () -> ExaModels.Optimizer(adapter)
    model = Model(optimizer_with_attributes(factory,
        "tol"=>1e-8,"print_level"=>MadNLP.ERROR,"max_iter"=>(kind == :limit ? 0 : 200),
        "linear_solver"=>MadNLP.MumpsSolver,"blas_num_threads"=>1))
    @variable(model,-2 <= x <= 2,start=0.5)
    if kind == :infeasible
        @objective(model,Min,x^2)
        @constraint(model,x^2+1 <= 0)
    elseif kind == :invalid
        @objective(model,Min,1/(x-0.5))
    else
        @objective(model,Min,(x-1)^2)
    end
    optimize!(model)
    return model,x
end

using ExaModels, MadNLP, SparseArrays
if "--gpu" in ARGS
    @eval using CUDA, CUDSS, MadNLPGPU
end
@testset "Owned ExaModels/MadNLP status adapter" begin
    adapter = ExaHVDC.MadNLPSolve()
    native = Base.get_extension(MadNLP,:MadNLPMOI)
    for status in instances(MadNLP.Status)
        @test ExaModels.termination_status_translator(adapter,status) == get(native._STATUS_CODES,status,JuMP.MOI.OTHER_ERROR)
        expected = status == MadNLP.SOLVE_SUCCEEDED ? JuMP.MOI.FEASIBLE_POINT :
            status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL ? JuMP.MOI.NEARLY_FEASIBLE_POINT :
            status == MadNLP.INFEASIBLE_PROBLEM_DETECTED ? JuMP.MOI.INFEASIBLE_POINT : JuMP.MOI.UNKNOWN_RESULT_STATUS
        @test ExaModels.result_status_translator(adapter,status) == expected
    end
    success,x = status_model(:success)
    @test termination_status(success) == JuMP.MOI.LOCALLY_SOLVED
    @test primal_status(success) == JuMP.MOI.FEASIBLE_POINT
    @test value(x) ≈ 1 atol=1e-6
    for (kind,status) in ((:limit,JuMP.MOI.ITERATION_LIMIT),(:infeasible,JuMP.MOI.LOCALLY_INFEASIBLE),(:invalid,JuMP.MOI.INVALID_MODEL))
        model,_ = status_model(kind)
        @test termination_status(model) == status
        @test primal_status(model) != JuMP.MOI.FEASIBLE_POINT
    end
end

@testset "Derivative comparison detects isolated errors" begin
    agree = ExaDerivativeChecks.agreement
    @test agree([1.0,1e8],[1.0,1e8])["passed"]
    @test !agree([1.0,1e8],[1.01,1e8])["passed"]
    @test !agree([NaN],[NaN])["passed"]
    @test !agree([Inf],[Inf])["passed"]
    @test !agree(zeros(2),zeros(3))["passed"]
    @test !agree(sparse([1],[2],[1.0],3,3),spzeros(3,3))["passed"]
    matrix = ExaDerivativeChecks.coordinate_matrix([(2,1),(1,2)],[1.0,2.0],2,2;symmetric=true)
    @test matrix[2,1] == 3.0
end

@testset "Small HVDC ExaModels validation" begin
    cpu_only = !("--gpu" in ARGS)
    mktempdir() do parent
        output = joinpath(parent,"results")
        @test run_exa_validation(["--output",output];cpu_only)
        report = TOML.parsefile(joinpath(output,"summary.toml"))
        @test report["gpu_validated"] == !cpu_only
        @test length(report["runs"]) == 2
        for entry in report["runs"]
            for mode in (cpu_only ? ("exa_cpu",) : ("exa_cpu","exa_gpu"))
                derivatives = TOML.parsefile(joinpath(output,entry["directory"],mode*"_derivatives.toml"))
                @test derivatives["passed"]
                @test derivatives["nonzero_constraint_multipliers"]
                @test length(derivatives["points"]) == 3
            end
        end
    end
    if cpu_only
        @test !any(id.name in ("CUDA","CUDSS","MadNLPGPU") for id in keys(Base.loaded_modules))
    end
end
