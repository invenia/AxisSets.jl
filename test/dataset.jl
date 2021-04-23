@testset "KeyedDataset" begin
    @testset "Construction" begin
        @testset "Invalid" begin
            # Test that you can construct an invalid dataset if you really want to
            @test_throws KeyAlignmentError KeyedDataset(
                OrderedSet(Pattern[(:__, :time)]),
                LittleDict(
                    (:val1,) => KeyedArray(rand(4); time=1:4),
                    (:val2,) => KeyedArray(rand(4); time=2:5),
                ),
            )

            ds = KeyedDataset(
                OrderedSet(Pattern[(:__, :time)]),
                LittleDict(
                    (:val1,) => KeyedArray(rand(4); time=1:4),
                    (:val2,) => KeyedArray(rand(4); time=2:5),
                ),
                false
            )
            @test_throws KeyAlignmentError validate(ds)
        end

        @testset "Empty" begin
            expected = KeyedDataset(OrderedSet{Pattern}(), LittleDict{Tuple, KeyedArray}())
            @test KeyedDataset() == expected

            patterns = Pattern[(:train, :_, :target)]
            expected = KeyedDataset(OrderedSet(patterns), LittleDict{Tuple, KeyedArray}())
            @test KeyedDataset(; constraints=patterns) == expected
        end

        @testset "KeyedArrays" begin
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

            @testset "Variable Keys" begin
                # Test construction with different key types and lengths
                ds = KeyedDataset(
                    "val1" => KeyedArray(
                        rand(4, 3, 2);
                        time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                        loc=1:3,
                        obj=[:a, :b],
                    ),
                    (:group1, 2) => KeyedArray(
                        rand(4, 3, 2) .+ 1.0;
                        time=DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                        loc=1:3,
                        obj=[:a, :b],
                    ),
                )

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj], dimnames(ds))
                # Test that we have a data dict entry for each value column
                @test issetequal([("val1",), (:group1, 2)], keys(ds.data))
            end
        end

        @testset "Flatten" begin
            # Technically this test is just showing how you can use the `flatten` function
            # to construct a KeyedDatasets from an existing nested structure of KeyedArrays.
            @testset "NamedTuples" begin
                # In this case, the resulting keys from flatten need to be symbols with
                # an `:_` delimiter
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
                ds = KeyedDataset(flatten(data)...)

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
                    [(:group1, :a), (:group1, :b), (:group2, :a), (:group2, :b)], keys(ds.data)
                )

                # Test an example where we pass a delimiter to flatten to a named tuple,
                # that we can use as kwargs...
                ds = KeyedDataset(; flatten(data, :_)...)

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
                    [(:group1_a,), (:group1_b,), (:group2_a,), (:group2_b,)], keys(ds.data)
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
                ds = KeyedDataset(flatten(data)...)

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], dimnames(ds))

                # Test that we successfully extracted the flattened pairs as tuples.
                @test issetequal(
                    [(:group1, :a), (:group1, :b), (:group2, :a), (:group2, :b)], keys(ds.data)
                )

                # We can still choose to flatten to a single symbol if we want.
                ds = KeyedDataset(flatten(data, :_)...)

                @test issetequal(
                    [(:group1_a,), (:group1_b,), (:group2_a,), (:group2_b,)], keys(ds.data)
                )

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
            end

            @testset "Mixed key types" begin
                # In this case, we have the option to having tuple keys which allows for
                # multi-dimensional style indexing.
                data = [
                    :group1 => [
                        "a" => KeyedArray(
                            rand(4, 3, 2);
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
                ]
                ds = KeyedDataset(flatten(data)...)

                # Test that we successfully extracted the dims
                @test issetequal([:time, :loc, :obj, :label], dimnames(ds))

                # Test that we successfully extracted the flattened pairs as tuples.
                @test issetequal(
                    [(:group1, "a"), (:group1, "b"), (:group2, "a"), (:group2, "b")], keys(ds.data)
                )
            end
        end
    end
    @testset "show" begin
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
            ),
        )

        # This is likely to change in the future, so we keep this test simple
        s = sprint(show, ds)
        @test startswith(s, "KeyedDataset")
        @test occursin("2 components", s)
        @test occursin("3 constraints", s)

        # Having unused constraints is fine since we might add them later.
        s = sprint(show, KeyedDataset(pairs(ds.data)...; constraints=Pattern[(:__, :foo)]))
        @test occursin("2 components", s)
        @test occursin("1 constraints", s)

        # While possible to construct it's rather pointless to have a dimension path
        # that can map to multiple constraints.
        bad_ds = KeyedDataset(
            pairs(ds.data)...;
            constraints=Pattern[
                (:__, :time),
                (:val1, :time),
            ],
        )
        @test_throws ArgumentError sprint(show, bad_ds)
    end

    @testset "dimpaths" begin
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
            ),
        )

        @test issetequal(
            dimpaths(ds),
            [
                (:val1, :time),
                (:val1, :loc),
                (:val1, :obj),
                (:val2, :time),
                (:val2, :loc),
                (:val2, :obj),
            ],
        )

        @test issetequal(dimpaths(ds, Pattern(:__, :time)), [(:val1, :time), (:val2, :time)])
    end

    @testset "dimnames" begin
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
                id=1:3,
                obj=[:a, :b],
            ),
        )
        @test issetequal(dimnames(ds), [:time, :loc, :obj, :id])
    end

    @testset "axiskeys" begin
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
                id=1:3,
                obj=[:a, :b],
            ),
        )

        @test issetequal(
            axiskeys(ds),
            [
                DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14),
                1:3,
                [:a, :b],
            ],
        )

        @test axiskeys(ds, (:val1, :time)) == DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14)
        @test unique(axiskeys(ds, :time)) == [DateTime(2021, 1, 1, 11):Hour(1):DateTime(2021, 1, 1, 14)]
    end

    @testset "validate" begin
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
                id=1:3,
                obj=[:a, :b],
            ),
        )

        @test validate(ds)
        @test validate(ds, Pattern(:__, :time))

        # Intentionally break the internal keys
        ds.data[(:val1,)] = KeyedArray(
            rand(4, 3, 2);
            time=DateTime(2021, 1, 2, 11):Hour(1):DateTime(2021, 1, 2, 14),
            loc=1:3,
            obj=[:a, :b],
        )

        @test_throws KeyAlignmentError validate(ds)
        @test_throws KeyAlignmentError validate(ds, Pattern(:__, :time))
        @test validate(ds, Pattern(:__, :obj))
    end
end
