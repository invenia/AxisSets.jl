using AxisSets: flatten

@testset "utils" begin
    @testset "flatten" begin
        @testset "Pairs" begin
            data = [
                :val1 => [:a1 => 1, :a2 => 2],
                :val2 => [:b1 => 11, :b2 => 22],
                :val3 => [111, 222],
                :val4 => 4.3,
            ]
            expected = [
                (:val1, :a1) => 1,
                (:val1, :a2) => 2,
                (:val2, :b1) => 11,
                (:val2, :b2) => 22,
                (:val3,) => [111, 222],
                (:val4,) => 4.3,
            ]
            @test flatten(data) == expected
        end

        @testset "Dict" begin
            data = Dict(
                :val1 => Dict(:a1 => 1, :a2 => 2),
                :val2 => Dict(:b1 => 11, :b2 => 22),
                :val3 => [111, 222],
                :val4 => 4.3,
            )
            expected = Dict(
                (:val1, :a1) => 1,
                (:val1, :a2) => 2,
                (:val2, :b1) => 11,
                (:val2, :b2) => 22,
                (:val3,) => [111, 222],
                (:val4,) => 4.3,
            )
            @test flatten(data) == expected
        end

        @testset "NamedTuple" begin
            data = (
                val1 = (a1 = 1, a2 = 2),
                val2 = (b1 = 11, b2 = 22),
                val3 = [111, 222],
                val4 = 4.3,
            )
            # Expected names for NamedTuple must be symbols. Default delimiter is `:_`
            expected = (
                val1_a1 = 1,
                val1_a2 = 2,
                val2_b1 = 11,
                val2_b2 = 22,
                val3 = [111, 222],
                val4 = 4.3,
            )
            @test flatten(data) == expected
        end
    end
end
