using Test
using LlamaFusion

@testset "LlamaFusion Basic Tests" begin
    
    @testset "Array Creation and Operations" begin
        # Test Array𝕍 creation
        A = Array𝕍([1.0, 2.0, 3.0], (3,), Dict{Symbol,Any}(:type => "test_vector"))
        B = Array𝕍([4.0, 5.0, 6.0], (3,), Dict{Symbol,Any}(:type => "test_vector"))
        
        @test A.data == [1.0, 2.0, 3.0]
        @test A.shape == (3,)
        @test A.mathematical_properties[:type] == "test_vector"
        
        # Test addition
        C = A + B
        @test C.data == [5.0, 7.0, 9.0]
        @test C.mathematical_properties[:operation] == :addition
    end
    
    @testset "Tensor Creation" begin
        # Test Tensor𝕋 creation
        T = Tensor𝕋([1.0, 2.0], (2,), nothing, true)
        @test T.data == [1.0, 2.0]
        @test T.requires_grad == true
        @test T.grad === nothing
    end
    
    @testset "Function Creation" begin
        # Test Function𝔽 creation
        f = Function𝔽(x -> x^2, [:square], Dict{Symbol,Any}(:domain => "real"))
        @test f.func(3.0) == 9.0
        @test :square in f.transformations
        @test f.mathematical_properties[:domain] == "real"
    end
    
    @testset "Mathematical Operators" begin
        A = Array𝕍([1.0, 2.0], (2,), Dict{Symbol,Any}())
        B = Array𝕍([3.0, 4.0], (2,), Dict{Symbol,Any}())
        
        # Test matrix multiplication
        A_2d = Array𝕍([1.0 2.0; 3.0 4.0], (2, 2), Dict{Symbol,Any}())
        B_2d = Array𝕍([5.0 6.0; 7.0 8.0], (2, 2), Dict{Symbol,Any}())
        
        C = A_2d * B_2d
        expected = [1.0*5.0+2.0*7.0 1.0*6.0+2.0*8.0; 3.0*5.0+4.0*7.0 3.0*6.0+4.0*8.0]
        @test C.data == expected
    end
    
    @testset "Progressive Syntax" begin
        # Test that help functions don't error
        A = Array𝕍([1, 2, 3], (3,), Dict{Symbol,Any}())
        @test_nowarn show_help(A, :beginner)
        @test_nowarn show_help(A, :intermediate) 
        @test_nowarn show_help(A, :expert)
    end
    
    @testset "Unicode Mathematical Symbols" begin
        # Test that Unicode symbols are exported and available
        @test isdefined(LlamaFusion, :∇)
        @test isdefined(LlamaFusion, :∂)
        @test isdefined(LlamaFusion, :∫)
        @test isdefined(LlamaFusion, :∑)
        @test isdefined(LlamaFusion, :∏)
        @test isdefined(LlamaFusion, :⊗)
        @test isdefined(LlamaFusion, :⊕)
        @test isdefined(LlamaFusion, :⊙)
    end
end