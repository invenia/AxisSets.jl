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

            # Test a nested singleton pair
            @test flatten("val1" => "a1" => 1) == [("val1", "a1") => 1]
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

            # Test that we can operate on keys that aren't symbols or strings
            data = Dict(
                DateTime(2021, 1, 1, 11) => Dict(1 => "a", 2 => "b"),
                DateTime(2021, 1, 1, 12) => Dict(1 => "x", 2 => "y"),
                DateTime(2021, 1, 1, 13) => [111, 222],
                DateTime(2021, 1, 1, 14) => 4.3,
            )

            expected = Dict(
                (DateTime(2021, 1, 1, 11), 1) => "a",
                (DateTime(2021, 1, 1, 11), 2) => "b",
                (DateTime(2021, 1, 1, 12), 1) => "x",
                (DateTime(2021, 1, 1, 12), 2) => "y",
                (DateTime(2021, 1, 1, 13),) => [111, 222],
                (DateTime(2021, 1, 1, 14),) => 4.3,
            )
            @test flatten(data) == expected
            @test flatten(expected) == expected

            expected = Dict(
                "2021-01-01T11:00:00_1" => "a",
                "2021-01-01T11:00:00_2" => "b",
                "2021-01-01T12:00:00_1" => "x",
                "2021-01-01T12:00:00_2" => "y",
                "2021-01-01T13:00:00" => [111, 222],
                "2021-01-01T14:00:00" => 4.3,
            )
            # Test that we can flatten that to a string
            @test flatten(data, "_") == expected
            @test flatten(expected, "_") == expected
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
            @test flatten(data, :_) == expected
            @test flatten(expected, :_) == expected

            # By default calling `flatten` on a named tuple without a delimiter will
            # return pairs
            expected = [
                (:val1, :a1) => 1,
                (:val1, :a2) => 2,
                (:val2, :b1) => 11,
                (:val2, :b2) => 22,
                (:val3,) => [111, 222],
                (:val4,) => 4.3,
            ]
            @test flatten(data) == expected
            @test flatten(expected) == expected
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
                    loc_obj=[Symbol(join((l, o), :_)) for l in 1:3, o in [:a, :b]][:],
                )
                @test flatten(A, (:loc, :obj), :_) == expected
            end

            @testset "head dims" begin
                expected = KeyedArray(
                    reshape(1:24, (12, 2));
                    time_loc=[Symbol(join((t, l), :_)) for t in dt, l in 1:3][:],
                    obj=[:a, :b],
                )
                @test flatten(A, (:time, :loc), :_) == expected
            end

            @test_throws ArgumentError flatten(A, (:time,), :_)
            @test_throws ArgumentError flatten(A, (:time, :obj), :_)
        end

        # TODO: Support group flatten operations
    end
end
