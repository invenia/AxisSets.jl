using AxisKeys
using AxisSets
using Dates
using Impute
using Missings
using ReadOnlyArrays
using Tables
using Test
using TimeZones

using AxisSets: Dataset

@testset "AxisSets.jl" begin
    @testset "Construction" begin
        @testset "KeyedArrays" begin
            ds = Dataset(
                :time, :loc, :obj;
                val1 = KeyedArray(
                    rand(25, 10, 3);
                    time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                    loc=1:10,
                    obj=[:a, :b, :c],
                ),
                val2 = KeyedArray(
                    rand(25, 10, 3) .+ 1.0;
                    time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                    loc=1:10,
                    obj=[:a, :b, :c],
                ),
            )

            # Test that we successfully extracted the dims
            @test :time in ds.dims
            @test :loc in ds.dims
            @test :obj in ds.dims

            # Test that we have a data dict entry for each value column
            @test :val1 in keys(ds.data)
            @test :val2 in keys(ds.data)
        end

        @testset "Tables" begin
            key_cols = Iterators.product(
                DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                1:10,
                [:a, :b, :c]
            )
            # We'll just start with a basic rowtable
            table = map(key_cols) do t
                vals = (t..., rand(), rand() + 1)
                return NamedTuple{(:time, :loc, :obj, :val1, :val2)}(vals)
            end |> vec

            @test Tables.isrowtable(table)

            # Test constructing AxisSet from table
            ds = Dataset(table, :time, :loc, :obj)

            # Test that we successfully extracted the dims
            @test :time in ds.dims
            @test :loc in ds.dims
            @test :obj in ds.dims

            # Test that we have a data dict entry for each value column
            @test :val1 in keys(ds.data)
            @test :val2 in keys(ds.data)
        end
    end

    @testset "ReadOnly" begin
        # Test that mutating data properties directly errors
        ds = Dataset(
            :time, :loc;
            val1 = KeyedArray(
                rand(25, 10, 3);
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
                obj=[:a, :b, :c],
            ),
            val2 = KeyedArray(
                rand(25, 10) .+ 1.0;
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
            ),
        )

        @test isa(ds.time, ReadOnlyArray)
        @test isa(ds.val1.time, ReadOnlyArray)

        @test_throws ErrorException ds.time[3] = DateTime(2020, 1, 1)
        @test_throws ErrorException ds.val1.time[3] = DateTime(2020, 1, 1)

        # Test that we can still modify non-shared keys
        ds.val1.obj[3] = :d
        @test ds.val1.obj[3] === :d
    end

    @testset "Filter Axis" begin
        ds = Dataset(
            :time, :loc, :obj;
            val1 = KeyedArray(
                allowmissing(rand(25, 10, 3));
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
                obj=[:a, :b, :c],
            ),
            val2 = KeyedArray(
                allowmissing(rand(25, 10) .+ 1.0);
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
            ),
        )

        # Add a couple missing values to test filtering across different dims
        ds.val1[3, 2, 3] = missing
        ds.val2[5, 1] = missing

        time_filtered = Impute.filter(ds; dims=:time)
        # We expect to have 2 timestamps removed from both components since we have missing
        # values at different times.
        @test size(time_filtered.val1) == (23, 10, 3)
        @test size(time_filtered.val2) == (23, 10)

        loc_filtered = Impute.filter(ds; dims=:loc)
        # We expect to have 2 locations removed from both components.
        @test size(loc_filtered.val1) == (25, 8, 3)
        @test size(loc_filtered.val2) == (25, 8)
    end

    @testset "Re-order Axis" begin
        ds = Dataset(
            :time, :obj;
            val1 = KeyedArray(
                allowmissing(rand(25, 3));
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                obj=[:a, :b, :c],
            ),
            val2 = KeyedArray(
                allowmissing(rand(25, 3) .+ 1.0);
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                obj=[:a, :b, :c],
            ),
        )

        AxisSets.permutekey!(ds, :obj, [1, 3, 2])
        @test ds.obj == [:a, :c, :b]
    end

    @testset "Mutate Axis Values" begin
        ds = Dataset(
            :time, :loc;
            val1 = KeyedArray(
                allowmissing(rand(25, 10, 3));
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
                obj=[:a, :b, :c],
            ),
            val2 = KeyedArray(
                allowmissing(rand(25, 10) .+ 1.0);
                time=DateTime(2021, 1, 1):Hour(1):DateTime(2021, 1, 2),
                loc=1:10,
            ),
        )

        AxisSets.remapkey!(dt -> ZonedDateTime(dt, tz"UTC"), ds, :time)
        @test eltype(ds.time) <: ZonedDateTime
        @test eltype(ds.val1.time) <: ZonedDateTime
        @test eltype(ds.val2.time) <: ZonedDateTime

        AxisSets.remapkey!(x -> x + 1, ds, :loc)
        @test ds.loc == 2:11
        @test ds.loc == 2:11
        @test ds.loc == 2:11
    end

    @testset "Merge and Split Axes" begin
        # Not quite sure how this should work yet.
        # Split seems easy, but maybe not that useful and merge seems potentially destructive.
    end

    @testset "Tables" begin
        # Test writing to a table (e.g., pretty_table)
    end

    @testset "Impute" begin
        # Other imputation methods
    end

    @testset "Add components" begin
        # Support adding and removing components with the dict like syntax
    end
end
