@testset "getproperty" begin
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
            obj=[:x, :y],
        );
        constraints=Pattern[(:__, :time), (:__, :loc)],
    )

    @test getproperty(ds, :constraints) == getfield(ds, :constraints)
    @test getproperty(ds, :data) == getfield(ds, :data)
    @test ds.val1 == ds.data[(:val1,)]
    @test ds.time == ds.data[(:val1,)].time
    @test_throws ArgumentError ds.obj
    @test_throws ArgumentError ds.foobar

    @testset "ReadOnly" begin
        # Test that mutating data properties directly errors
        @test isa(ds.time, ReadOnlyArray)
        @test isa(ds.val1.time, ReadOnlyArray)

        @test_throws ErrorException ds.time[3] = DateTime(2020, 1, 1)
        @test_throws ErrorException ds.val1.time[3] = DateTime(2020, 1, 1)

        # Test that we can still modify non-shared keys
        ds.val1.obj[2] = :d
        @test ds.val1.obj[2] === :d
    end
end

@testset "getindex" begin
    @testset "depth = 1" begin
        ds = KeyedDataset(
            :a => KeyedArray(
                allowmissing(rand(4, 3, 2));
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                loc=1:3,
                obj=[:a, :b],
            ),
            :b => KeyedArray(
                rand(4, 2);
                time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                label=["x", "y"],
            ),
        )

        @test ds[:a] == ds[(:a,)]
        @test isa(ds[:a], KeyedArray)
    end

    @testset "depth > 1" begin
        ds = KeyedDataset(
            flatten([
                :group1 => [
                    :a => KeyedArray(
                        allowmissing(rand(4, 3, 2));
                        time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                        loc=1:3,
                        obj=[:a, :b],
                    ),
                ],
                :group2 => [
                    :b => KeyedArray(
                        rand(4, 2) .+ 1.0;
                        time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                        label=["x", "y"],
                    ),
                ],
            ])...
        )

        grp1a = ds[(:group1, :a)]
        @test isa(grp1a, KeyedArray)

        grp2b = ds[(:group2, :b)]
        @test isa(grp2b, KeyedArray)

        @test_throws KeyError ds[(:group1, :)]
        @test_throws KeyError ds[:group1]
        @test_throws KeyError ds[:a]
    end
end

@testset "setindex!" begin
    ds = KeyedDataset(
        :a => KeyedArray(
            allowmissing(rand(4, 3, 2));
            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
            loc=1:3,
            obj=[:a, :b],
        ),
        :b => KeyedArray(
            rand(4, 2);
            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
            label=["x", "y"],
        ),
    )

    init_constraints = deepcopy(ds.constraints)
    @test length(init_constraints) == 4

    ds[:c] = KeyedArray(
        rand(4, 2);
        time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
        label=["x", "y"],
    )
    @test ds.constraints == init_constraints

    ds[:c] = KeyedArray(rand(5); id=100:100:500)
    @test ds.constraints != init_constraints
    @test length(ds.constraints) == 5

    # Test validation error if we try to add a component with a mismatched dimension
    # relative to our existing dimensions / constraints
    @test_throws KeyAlignmentError setindex!(ds, KeyedArray(rand(2); id=[1, 2]), :d)
end

@testset "lookup" begin
    ds = KeyedDataset(
        flatten([
            :group1 => [
                :a => KeyedArray(
                    allowmissing(rand(4, 3, 2));
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                ),
                :b => KeyedArray(
                    rand(4, 2);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    label=["x", "y"],
                )
            ],
            :group2 => [
                :a => KeyedArray(
                    rand(4, 3, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                ),
                :b => KeyedArray(
                    rand(4, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    label=["x", "y"],
                )
            ]
        ])...
    )

    a = ds(:_, :a)
    @test isa(a, KeyedDataset)
    @test issetequal([(:group1, :a), (:group2, :a)], keys(a.data))

    grp1 = ds(:group1, :__)
    @test isa(grp1, KeyedDataset)
    @test issetequal([(:group1, :a), (:group1, :b)], keys(grp1.data))

    @test_throws KeyError ds(:group1)
    @test_throws KeyError ds(:a)

    # Test custom filter
    custom = ds(k -> any(startswith("gr"), string.(k)))
    @test custom == ds
end

@testset "mixed key types" begin
    ds = KeyedDataset(
        flatten([
            :group1 => [
                "a" => KeyedArray(
                    allowmissing(rand(4, 3, 2));
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                ),
                "b" => KeyedArray(
                    rand(4, 2);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    label=["x", "y"],
                )
            ],
            :group2 => [
                "a" => KeyedArray(
                    rand(4, 3, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                ),
                "b" => KeyedArray(
                    rand(4, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    label=["x", "y"],
                )
            ]
        ])...
    )

    grp1a = ds[(:group1, "a")]
    @test isa(grp1a, KeyedArray)

    grp2b = ds[(:group2, "b")]
    @test isa(grp2b, KeyedArray)

    a = ds(:_, "a")
    @test isa(a, KeyedDataset)
    @test issetequal([(:group1, "a"), (:group2, "a")], keys(a.data))

    grp1 = ds(:group1, :__)
    @test isa(grp1, KeyedDataset)
    @test issetequal([(:group1, "a"), (:group1, "b")], keys(grp1.data))
end
