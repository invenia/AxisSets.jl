using AxisSets: flatten

@testset "utils" begin
    @testset "flatten" begin
        @testset "Pairs{Symbol}" begin
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

            delimited = [
                :val1_a1 => 1,
                :val1_a2 => 2,
                :val2_b1 => 11,
                :val2_b2 => 22,
                :val3 => [111, 222],
                :val4 => 4.3,
            ]
            @test flatten(data, :_) == delimited
        end

        @testset "Pairs{String}" begin
            data = [
                "val1" => ["a1" => 1, "a2" => 2],
                "val2" => ["b1" => 11, "b2" => 22],
                "val3" => [111, 222],
                "val4" => 4.3,
            ]
            expected = [
                ("val1", "a1") => 1,
                ("val1", "a2") => 2,
                ("val2", "b1") => 11,
                ("val2", "b2") => 22,
                ("val3",) => [111, 222],
                ("val4",) => 4.3,
            ]
            @test flatten(data) == expected

            delimited = [
                "val1_a1" => 1,
                "val1_a2" => 2,
                "val2_b1" => 11,
                "val2_b2" => 22,
                "val3" => [111, 222],
                "val4" => 4.3,
            ]
            @test flatten(data, "_") == delimited
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
            # Expected names for NamedTuple must be symbols. Default delimiter is `:⁻`
            expected = (
                val1⁻a1 = 1,
                val1⁻a2 = 2,
                val2⁻b1 = 11,
                val2⁻b2 = 22,
                val3 = [111, 222],
                val4 = 4.3,
            )
            @test flatten(data) == expected
        end

        @testset "KeyedArray" begin
            dt = DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14)
            A = KeyedArray(
                reshape(1:24, (4, 3, 2));
                time=dt,
                loc=1:3,
                obj=[:a, :b],
            )

            @testset "tail dims" begin
                expected = KeyedArray(
                    reshape(1:24, (4, 6));
                    time=dt,
                    locᵡobj=[Symbol(join((l, o), :ᵡ)) for l in 1:3, o in [:a, :b]][:],
                )
                @test flatten(A, (:loc, :obj)) == expected
            end

            @testset "head dims" begin
                expected = KeyedArray(
                    reshape(1:24, (12, 2));
                    timeᵡloc=[Symbol(join((t, l), :ᵡ)) for t in dt, l in 1:3][:],
                    obj=[:a, :b],
                )
                @test flatten(A, (:time, :loc)) == expected
            end

            @test_throws ArgumentError flatten(A, (:time,))
            @test_throws ArgumentError flatten(A, (:time, :obj))
        end

        # TODO: Support group flatten operations
    end
end
