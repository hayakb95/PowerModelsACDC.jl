using Test, TOML
include("run.jl") # Includes the helpers; the CLI guard prevents GPU imports/solves.

@testset "Validation launcher and portable records (CPU only)" begin
    mktempdir() do root
        output = joinpath(root,"output")
        opts = validation_options(["--output",output])
        @test opts.comparison == "strict"
        @test length(opts.paths) == 2
        @test validation_options(["--output",output,"--large"]).comparison == "strict"
        @test length(validation_options(["--output",output,"--large"]).paths) == 3
        @test validation_options(["--output",output,"--comparison","verified-sharing"]).comparison == "verified-sharing"
        @test_throws ErrorException validation_options(["--output",root])
        @test_throws ErrorException validation_options(["--output",output,"--comparison","bad"])
        @test_throws ErrorException validation_options(["--output",output,"--large","case5_acdc.m"])
        @test_throws ErrorException validation_options(["--output",output,"--unknown"])
        @test_throws ErrorException validation_options(["--output"])
        @test_throws ErrorException validation_options(String[])
        yes,no=Dict("passed"=>true),Dict("passed"=>false)
        @test !GPUValidationRecords.accepted(yes,yes,no,yes,"strict")
        @test GPUValidationRecords.accepted(yes,yes,no,yes,"verified-sharing")
        @test !GPUValidationRecords.accepted(no,yes,no,yes,"verified-sharing")
        @test !GPUValidationRecords.accepted(yes,no,no,yes,"verified-sharing")
        @test !GPUValidationRecords.accepted(yes,yes,yes,no,"verified-sharing")
        @test_throws ErrorException GPUValidationRecords.accepted(yes,yes,yes,yes,"bad")

        casefile = pkgdir(PowerModelsACDC,"test","data","case5_acdc.m")
        result,pm = solve_hvdc_example(;gpu=false,casefile,matched_treatments=true)
        record = GPUValidationRecords.capture(result,pm,casefile)
        @test record["passed"]
        @test record["mode"] == "cpu"
        path = joinpath(root,"cpu.toml")
        GPUValidationRecords.save_record(path,record)
        @test TOML.parsefile(path) == record
        @test_throws ErrorException GPUValidationRecords.save_record(path,record)
        # Replay uses the same stored point on both sides; this is a record-format
        # test, not GPU validation. No CUDA package is loaded by this script.
        report_path = joinpath(root,"replay.toml")
        @test VerifiedSharing.compare_sharing_files([path,path,casefile,report_path])
        @test TOML.parsefile(report_path)["strict_passed"]
        @test_throws ErrorException VerifiedSharing.compare_sharing_files([path,path,casefile,report_path])
        @test !any(id.name in ("CUDA","CUDSS","MadNLPGPU","ExaModels") for id in keys(Base.loaded_modules))
    end
end
