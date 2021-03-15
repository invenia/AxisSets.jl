@testset "map" begin
    ds = KeyedDataset(
        flatten([
            :g1 => [
                :a => KeyedArray(zeros(3); time=1:3),
                :b => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
            ],
            :g2 => [
                :a => KeyedArray(ones(3); time=1:3),
                :b => KeyedArray(zeros(3, 2); time=1:3, loc=[:x, :y]),
            ]
        ])...
    );

    # Since all of the tests are reusing `ds` we can assum it isn't being mutated.
    # Maybe there's a better way to test the combinations, but this seemed pretty terse?

    # Test mapping over every component
    r = map(a -> a .+ 1.0, ds)
    @test mean(r[(:g1, :a)]) == mean(r[(:g2, :b)]) == 1.0
    @test mean(r[(:g1, :b)]) == mean(r[(:g2, :a)]) == 2.0

    # Test mapping over components
    r = map(a -> a .+ 1.0, ds, (:g1, :__))
    @test mean(r[(:g2, :b)]) == 0.0
    @test mean(r[(:g1, :a)]) == mean(r[(:g2, :a)]) == 1.0
    @test mean(r[(:g1, :b)]) == 2.0

    # Test mapping over more complex patterns
    r = map(a -> a .+ 1.0, ds, (:__, :a, :_))
    @test mean(r[(:g2, :b)]) == 0.0
    @test mean(r[(:g1, :a)]) == mean(r[(:g1, :b)]) == 1.0
    @test mean(r[(:g2, :a)]) == 2.0

    # Test mapping just dims
    r = map(a -> a .+ 1.0, ds, (:__, :loc))
    @test mean(r[(:g1, :a)]) == 0.0
    @test mean(r[(:g2, :b)]) == mean(r[(:g2, :a)]) == 1.0
    @test mean(r[(:g1, :b)]) == 2.0

    # Test mapping over component and dim pattern
    r = map(a -> a .+ 1.0, ds, (:g1, :__, :loc))
    @test mean(r[(:g1, :a)]) == mean(r[(:g2, :b)]) == 0.0
    @test mean(r[(:g2, :a)]) == 1.0
    @test mean(r[(:g1, :b)]) == 2.0
end

@testset "mapslices" begin
    ds = KeyedDataset(
        flatten([
            :g1 => [
                :a => KeyedArray(zeros(3); time=1:3),
                :b => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
            ],
            :g2 => [
                :a => KeyedArray(ones(3); time=1:3),
                :b => KeyedArray(zeros(3, 2); time=1:3, loc=[:x, :y]),
            ]
        ])...
    )


    r = mapslices(sum, ds; dims=:time)
    expected = [
        (:g1, :a) => [0.0],
        (:g1, :b) => [3.0 3.0],
        (:g2, :a) => [3.0],
        (:g2, :b) => [0.0 0.0],
    ]
    for (k, v) in expected
        @test v == r[k]
    end

    r = mapslices(sum, ds, (:__, :b, :_); dims=:loc)
    @test r == mapslices(sum, ds; dims=:loc)
    expected = [
        (:g1, :a) => zeros(3),
        (:g1, :b) => fill(2.0, (3, 1)),
        (:g2, :a) => ones(3),
        (:g2, :b) => zeros((3, 1)),
    ]
    for (k, v) in expected
        @test v == r[k]
    end

    # Reducing over time will violate our :time dimension constraint
    @test_throws ArgumentError mapslices(sum, ds, (:__, :b, :_); dims=:time)
    # Dimension doesn't exist in the selection
    @test_throws ArgumentError mapslices(sum, ds, (:__, :b, :_); dims=:foo)
end

@testset "merge" begin
    # Test destructive merging of valid and invalid datasets
    @testset "additive" begin
        ds1 = KeyedDataset(
            :val1 => KeyedArray(
                rand(4, 3, 2);
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            )
        )

        ds2 = KeyedDataset(
            :val2 => KeyedArray(
                rand(4, 3, 2) .+ 1.0;
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            )
        )
        ds = merge(ds1, ds2)
        @test issetequal(keys(ds.data), [(:val1,), (:val2,)])
    end
    @testset "replace" begin
        ds1 = KeyedDataset(
            :val1 => KeyedArray(
                rand(4, 3, 2);
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            )
        )

        ds2 = KeyedDataset(
            :val1 => KeyedArray(
                rand(4, 3, 2) .+ 1.0;
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            )
        )
        ds = merge(ds1, ds2)
        @test issetequal(keys(ds.data), [(:val1,)])
        @test all(x -> x >= 1.0, ds.val1)
    end
    @testset "key mismatch" begin
        ds1 = KeyedDataset(
            :val1 => KeyedArray(
                rand(4, 3, 2);
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            )
        )

        ds2 = KeyedDataset(
            :val2 => KeyedArray(
                rand(4, 2, 2) .+ 1.0;
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:2,
                obj=[:a, :b],
            )
        )
        @test_throws ArgumentError merge(ds1, ds2)
    end
end

@testset "rekey" begin
    ds = KeyedDataset(
        :val1 => KeyedArray(
            rand(4, 3, 2);
            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
            loc=1:3,
            obj=[:a, :b],
        ),
        :val2 => KeyedArray(
            rand(4, 3, 2) .+ 1.0;
            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
            loc=1:3,
            obj=[:a, :b],
        );
    )

    r = AxisSets.rekey(k -> ZonedDateTime.(k, tz"UTC"), ds, :time)
    @test eltype(r.time) <: ZonedDateTime
    @test eltype(r.val1.time) <: ZonedDateTime
    @test eltype(r.val2.time) <: ZonedDateTime

    r = AxisSets.rekey(k -> k .+ 1, ds, (:__, :loc))
    @test r.loc == 2:4
    @test r.loc == 2:4
    @test r.loc == 2:4
end
