@testset "Dataset" begin
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
                ),
            )

            # Test that we successfully extracted the dims
            @test issetequal([:time, :loc, :obj], dimnames(ds))

            # Test generated constraints
            @test constraintmap(ds) == LittleDict(
                Pattern((:__, :time)) => Set([(:val1, :time), (:val2, :time)]),
                Pattern((:__, :loc)) => Set([(:val1, :loc), (:val2, :loc)]),
                Pattern((:__, :obj)) => Set([(:val1, :obj), (:val2, :obj)]),
            )

            # Test that we have a data dict entry for each value column
            @test issetequal([(:val1,), (:val2,)], keys(ds.data))
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
                ds = Dataset(; flatten(data)...)

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], dimnames(ds))

                # Test generated constraints
                @test constraintmap(ds) == LittleDict(
                    Pattern((:__, :time)) => Set([
                        (:group1, :a, :time),
                        (:group1, :b, :time),
                        (:group2, :a, :time),
                        (:group2, :b, :time),
                    ]),
                    Pattern((:__, :loc)) => Set([(:group1, :a, :loc), (:group2, :a, :loc)]),
                    Pattern((:__, :obj)) => Set([(:group1, :a, :obj), (:group2, :a, :obj)]),
                    Pattern((:__, :label)) => Set([
                        (:group1, :b, :label),
                        (:group2, :b, :label),
                    ]),
                )

                # Test that we successfully extracted the flattened kwargs.
                @test issetequal(
                    [(:group1, :a), (:group1, :b), (:group2, :a), (:group2, :b)], keys(ds)
                )

                # Test an example where we don't use the default delimiter
                ds = Dataset(; flatten(data, :_)...)

                @test constraintmap(ds) == LittleDict(
                    Pattern((:__, :time)) => Set([
                        (:group1_a, :time),
                        (:group1_b, :time),
                        (:group2_a, :time),
                        (:group2_b, :time),
                    ]),
                    Pattern((:__, :loc)) => Set([(:group1_a, :loc), (:group2_a, :loc)]),
                    Pattern((:__, :obj)) => Set([(:group1_a, :obj), (:group2_a, :obj)]),
                    Pattern((:__, :label)) => Set([(:group1_b, :label), (:group2_b, :label)]),
                )

                @test issetequal(
                    [(:group1_a,), (:group1_b,), (:group2_a,), (:group2_b,)], keys(ds)
                )
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
                ds = Dataset(flatten(data)...)

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], dimnames(ds))

                # Test that we successfully extracted the flattened pairs as tuples.
                @show keys(ds)
                @test issetequal(
                    [(:group1, :a), (:group1, :b), (:group2, :a), (:group2, :b)], keys(ds)
                )

                # We can still choose to flatten to a single symbol if we want.
                ds = Dataset(flatten(data, :_)...)
                @show keys(ds)
                @test issetequal(
                    [(:group1_a,), (:group1_b,), (:group2_a,), (:group2_b,)], keys(ds)
                )
            end
        end
    end
end
