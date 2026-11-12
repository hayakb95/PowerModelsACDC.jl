using Test, CUDA, CUDSS, MadNLPGPU
include("opf.jl")

# This is an explicit GPU suite: unavailable CUDA is a failure, not a skipped pass.
CUDA.functional() || error("Run this suite on a CUDA-capable GPU")
CUDA.allowscalar(false)

@testset "HVDC OPF with GPU solver" begin
    for case in ("case5_acdc.m", "case5_lcc.m")
        @testset "$case" begin
            path = pkgdir(PowerModelsACDC, "test", "data", case)
            cpu, cpu_pm = solve_hvdc_example(; gpu=false, casefile=path)
            gpu, gpu_pm = solve_hvdc_example(; gpu=true, casefile=path)
            for (result, pm) in ((cpu, cpu_pm), (gpu, gpu_pm))
                @test result["termination_status"] == JuMP.MOI.LOCALLY_SOLVED
                @test result["primal_status"] == JuMP.MOI.FEASIBLE_POINT
                @test maximum(values(JuMP.primal_feasibility_report(pm.model; atol=0.0)); init=0.0) <= 1e-6
                for table in ("busdc", "branchdc", "convdc")
                    @test !isempty(result["solution"][table])
                    @test all(isfinite(v) for row in values(result["solution"][table]) for v in values(row) if v isa Number)
                end
            end
            @test gpu["objective"] ≈ cpu["objective"] rtol=1e-6
            raw = JuMP.unsafe_backend(gpu_pm.model)
            @test raw.nlp.meta.x0 isa CUDA.CuArray
            @test MadNLP.full(raw.solver.x) isa CUDA.CuArray
            @test raw.solver.kkt.linear_solver isa MadNLPGPU.CUDSSSolver
            @test raw.solver.kkt.linear_solver.tril isa CUDA.CUSPARSE.CuSparseMatrixCSC
            @test raw.nlp.inner.meta.x0 isa Vector{Float64}
        end
    end
end
