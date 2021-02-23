using AxisSets: Dataset

@testset "Datasets" begin
    @testset "Construction" begin
        @testset "KeyedArrays" begin
            ds = Dataset(
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
                dims=(:time, :loc, :obj),
            )

            # Test that we successfully extracted the dims
            @test issetequal([:time, :loc, :obj], ds.dims)

            # Test that we have a data dict entry for each value column
            @test issetequal([:val1, :val2], keys(ds.data))
        end

        @testset "Tables" begin
            key_cols = Iterators.product(
                DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                1:3,
                [:a, :b]
            )
            # We'll just start with a basic rowtable
            table = map(key_cols) do t
                vals = (t..., rand(), rand() + 1)
                return NamedTuple{(:time, :loc, :obj, :val1, :val2)}(vals)
            end |> vec

            @test Tables.isrowtable(table)

            # Test constructing AxisSet from table
            ds = Dataset(table; dims=(:time, :loc, :obj))

            # Test that we successfully extracted the dims
            @test issetequal([:time, :loc, :obj], ds.dims)

            # Test that we have a data dict entry for each value column
            @test issetequal([:val1, :val2], keys(ds.data))
        end

        @testset "Flatten" begin
            # Technically this test is just showing how you can use the `flatten` function
            # to construct a Datasets from an existing nested structure of KeyedArrays.
            @testset "NamedTuples" begin
                # In this case, the resulting keys from flatten need to be symbols with
                # an `:ᵡ` delimiter
                data = (
                    group1 = (
                        a = KeyedArray(
                            rand(4, 3, 2);
                            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                            loc=1:3,
                            obj=[:a, :b],
                        ),
                        b = KeyedArray(
                            rand(4, 2);
                            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                            label=["x", "y"],
                        )
                    ),
                    group2 = (
                        a = KeyedArray(
                            rand(4, 3, 2) .+ 1.0;
                            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                            loc=1:3,
                            obj=[:a, :b],
                        ),
                        b = KeyedArray(
                            rand(4, 2) .+ 1.0;
                            time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                            label=["x", "y"],
                        )
                    )
                )
                ds = Dataset(; dims=(:time, :loc, :obj, :label), flatten(data)...)

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], ds.dims)

                # Test that we successfully extracted the flattened kwargs.
                @test issetequal([:group1ᵡa, :group1ᵡb, :group2ᵡa, :group2ᵡb], keys(ds.data))
            end

            @testset "Pairs" begin
                # In this case, we have the option to having tuple keys which allows for
                # multi-dimensional style indexing.
                data = [
                    :group1 => [
                        :a => KeyedArray(
                            rand(4, 3, 2);
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
                ]
                ds = Dataset(flatten(data)...; dims=(:time, :loc, :obj, :label))

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], ds.dims)

                # Test that we successfully extracted the flattened pairs as tuples.
                @test issetequal(
                    [(:group1, :a), (:group1, :b), (:group2, :a), (:group2, :b)],
                    keys(ds.data)
                )

                # We can still choose to flatten to a single symbol if we want.
                ds = Dataset(flatten(data, :_)...; dims=(:time, :loc, :obj, :label))
                @test issetequal([:group1_a, :group1_b, :group2_a, :group2_b], keys(ds.data))
            end
        end
    end

    @testset "Indexing" begin
        data = [
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
        ]
        ds = Dataset(flatten(data)...; dims=(:time, :loc, :obj, :label))

        grp1a = ds[(:group1, :a)]
        @test isa(grp1a, KeyedArray)

        A = ds[:a]
        @test isa(A, Dataset)
        @test issetequal([(:group1, :a), (:group2, :a)], keys(A.data))
        # Test that key mutation operations on the returned dataset will fail
        # Add a couple missing values to test filtering across different dims
        A[(:group1, :a)][3, 2, 2] = missing

        # Hmm, we probably want this to error given that the axes are ReadOnlyArrays,
        # but slicing operations are still valid cause they make a copy.
        time_filtered = Impute.filter(A; dims=:time)
        @test length(time_filtered.time) == 3
    end

    @testset "ReadOnly" begin
        # Test that mutating data properties directly errors
        ds = Dataset(
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
            dims=(:time, :loc),
        )

        @test isa(ds.time, ReadOnlyArray)
        @test isa(ds.val1.time, ReadOnlyArray)

        @test_throws ErrorException ds.time[3] = DateTime(2020, 1, 1)
        @test_throws ErrorException ds.val1.time[3] = DateTime(2020, 1, 1)

        # Test that we can still modify non-shared keys
        ds.val1.obj[2] = :d
        @test ds.val1.obj[2] === :d
    end

    @testset "Re-order Axis" begin
        ds = Dataset(
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
            dims=(:time, :loc, :obj),
        )

        AxisSets.permutekey!(ds, :obj, [2, 1])
        @test ds.obj == [:b, :a]
    end

    @testset "Mutate Axis Values" begin
        ds = Dataset(
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
            dims=(:time, :loc, :obj),
        )

        AxisSets.remapkey!(dt -> ZonedDateTime(dt, tz"UTC"), ds, :time)
        @test eltype(ds.time) <: ZonedDateTime
        @test eltype(ds.val1.time) <: ZonedDateTime
        @test eltype(ds.val2.time) <: ZonedDateTime

        AxisSets.remapkey!(x -> x + 1, ds, :loc)
        @test ds.loc == 2:4
        @test ds.loc == 2:4
        @test ds.loc == 2:4
    end

    @testset "Merge" begin
        # Test destructive merging of valid and invalid datasets
        @testset "additive" begin
            ds1 = Dataset(
                :val1 => KeyedArray(
                    rand(4, 3, 2);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                )
            )

            ds2 = Dataset(
                :val2 => KeyedArray(
                    rand(4, 3, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                )
            )
            ds = merge(ds1, ds2)
            @test issetequal(keys(ds), [:val1, :val2])
        end
        @testset "replace" begin
            ds1 = Dataset(
                :val1 => KeyedArray(
                    rand(4, 3, 2);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                )
            )

            ds2 = Dataset(
                :val1 => KeyedArray(
                    rand(4, 3, 2) .+ 1.0;
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                )
            )
            ds = merge(ds1, ds2)
            @test issetequal(keys(ds), [:val1])
            @test all(x -> x >= 1.0, ds.val1)
        end
        @testset "key mismatch" begin
            ds1 = Dataset(
                :val1 => KeyedArray(
                    rand(4, 3, 2);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                )
            )

            ds2 = Dataset(
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

    @testset "Join" begin
        # TODO: I'm not sure yet, but we might want some sort of non-destructive join
        # operation that can do right, left, inner or outer joins on axis keys.
    end

    @testset "Tables" begin
        key_cols = Iterators.product(
            DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
            1:3,
            [:a, :b]
        )
        # We'll just start with a basic rowtable
        table = map(key_cols) do t
            vals = (t..., rand(), rand() + 1)
            return NamedTuple{(:time, :loc, :obj, :val1, :val2)}(vals)
        end |> vec

        @test Tables.isrowtable(table)
        # Construct our Dataset

        # Test constructing AxisSet from table
        ds = Dataset(table; dims=(:time, :loc, :obj))

        # Test that we successfully extracted the dims
        @test issetequal([:time, :loc, :obj], ds.dims)

        # Test that we have a data dict entry for each value column
        @test issetequal([:val1, :val2], keys(ds.data))

        # Test that the dataset is also a table
        @test Tables.rowaccess(ds)
        @test Tables.columnaccess(ds)

        # Just compare the set of rows as we don't guarantee ordering.
        @test issetequal(Tables.rows(ds), table)
        # Add a test for cases where:
        # 1. the dataset doesn't describe a single dataset and should errors
        # 2. the dataset has components on a subset of shared dimensions (just :time x :loc)
    end

    @testset "Impute" begin
        @testset "Filter Axis" begin
            ds = Dataset(
                :val1 => KeyedArray(
                    allowmissing(rand(4, 3, 2));
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                    obj=[:a, :b],
                ),
                :val2 => KeyedArray(
                    allowmissing(rand(4, 3) .+ 1.0);
                    time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                    loc=1:3,
                );
                dims=(:time, :loc),
            )

            # Add a couple missing values to test filtering across different dims
            ds.val1[3, 2, 1] = missing
            ds.val2[2, 1] = missing

            time_filtered = Impute.filter(ds; dims=:time)
            # We expect to have 2 timestamps removed from both components since we have missing
            # values at different times.
            @test size(time_filtered.val1) == (2, 3, 2)
            @test size(time_filtered.val2) == (2, 3)

            loc_filtered = Impute.filter(ds; dims=:loc)
            # We expect to have 2 locations removed from both components.
            @test size(loc_filtered.val1) == (4, 1, 2)
            @test size(loc_filtered.val2) == (4, 1)
        end
        # Other imputation methods
    end

    @testset "Add components" begin
        # Support adding and removing components with the dict like syntax
        # I'm not entirely sold that we need this if we have merge operations?
        # How would this operation compare to a potential `join` operation that can handle
        # right, left, inner or outer joins on axis keys?
    end
end
