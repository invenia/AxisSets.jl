@testset "FeatureTransforms" begin

    M1 = [0.0 1.0; 1.0 2.0; -0.5 0.0]
    M2 = [-2.0 4.0; 3.0 2.0; -1.0 -1.0]
    M3 = [0.0 1.0; -1.0 0.5; -0.5 0.0]
    M4 = [0.5 -1.0; -5.0 -2.0; 0.0 1.0]

    ds = KeyedDataset(
        flatten([
            :train => [
                :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
            ],
            :predict => [
                :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
            ]
        ])...
    )

    @testset "transform" begin
        @test is_transformable(ds)
    end

    @testset "OneToOne" begin
        T = FakeOneToOneTransform()

        @testset "default applies to all components" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 2); time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 2); time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using pattern" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 2); time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 2); time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, (:_, :price, :_))

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using symbol" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using dims" begin
            # replaces the first :loc column with ones(...)
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 1); time=1:3, loc=[:x]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 1); time=1:3, loc=[:x]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc; dims=2, inds=[1])

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end
    end

    @testset "OneToMany" begin
        T = FakeOneToManyTransform()

        @testset "default applies to all components" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using pattern" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, (:_, :price, :_))

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using symbol" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using dims" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :x]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :x]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc; dims=2, inds=[1])

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end
    end

    @testset "ManyToOne" begin
        T = FakeManyToOneTransform()

        @testset "default applies to all components" begin
            expected = KeyedDataset(
                # ideally we would drop the :time constraint when it gets reduced
                OrderedSet(Pattern[(:__, :time), (:__, :loc), (:__, :id)]),
                LittleDict(flatten([
                    :train => [
                        :load => KeyedArray(ones(2); loc=[:x, :y]),
                        :price => KeyedArray(ones(2); id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(2); loc=[:x, :y]),
                        :price => KeyedArray(ones(2); id=[:a, :b]),
                    ]
                ])...)
            )

            r = FeatureTransforms.apply(ds, T; dims=:time)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test_broken !isequal(ds, expected)  # same problem as above
        end

        @testset "using pattern" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(2); id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(2); id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, (:_, :price, :_); dims=:time)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test_broken !isequal(ds, expected)  # same as above
        end

        @testset "using symbol" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(2); loc=[:x, :y]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(2); loc=[:x, :y]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc; dims=:time)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test_broken !isequal(ds, expected)
        end
    end

    # Note: There are no ManyToMany transforms implemented just yet
    @testset "ManyToMany" begin
        T = FakeManyToManyTransform()

        @testset "default applies to all components" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using pattern" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(ones(3, 4); time=1:3, id=[:a, :b, :a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, (:_, :price, :_))

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using symbol" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 4); time=1:3, loc=[:x, :y, :x, :y]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc)

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end

        @testset "using dims" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :x]),
                        :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(ones(3, 2); time=1:3, loc=[:x, :x]),
                        :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply(ds, T, :loc; dims=2, inds=[1])

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test !isequal(ds, expected)
        end
    end

end
