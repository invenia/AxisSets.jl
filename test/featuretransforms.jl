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

    # TODO: use fake Transforms
    @testset "apply OneToOne" begin
        p = Power(2)

        @testset "apply" begin
            @testset "one feature" begin
                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M2.^2; time=1:3, id=[:a, :b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M4.^2; time=1:3, id=[:a, :b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply(ds, p, (:_, :price, :_))

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)
            end

            @testset "all features" begin
                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1.^2; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M2.^2; time=1:3, id=[:a, :b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3.^2; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M4.^2; time=1:3, id=[:a, :b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply(ds, p; dims=:)

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)
            end

            @testset "inds" begin
                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(hcat((M2.^2)[:, 2]); time=1:3, id=[:b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(hcat((M4.^2)[:, 2]); time=1:3, id=[:b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply(ds, p, (:_, :price, :_); dims=:id, inds=[2])

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)
            end
        end

        @testset "apply!" begin
            expected = KeyedDataset(
                flatten([
                    :train => [
                        :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(M2.^2; time=1:3, id=[:a, :b]),
                    ],
                    :predict => [
                        :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                        :price => KeyedArray(M4.^2; time=1:3, id=[:a, :b]),
                    ]
                ])...
            )

            r = FeatureTransforms.apply!(ds, p, (:_, :price, :_))

            @test r isa KeyedDataset
            @test isequal(r, expected)
            @test isequal(ds, expected)
        end

        @testset "apply_append" begin
            @testset "inner" begin
                M2_cat = cat(M2, M2.^2, dims=2)
                M4_cat = cat(M4, M4.^2, dims=2)
                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M2_cat; time=1:3, id=[:a, :b, :a, :b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M4_cat; time=1:3, id=[:a, :b, :a, :b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply_append(
                    ds, p, (:_, :price, :_);
                    inner=true, append_dim=2
                )

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)
            end

            @testset "outer" begin
                M2_cat = cat(M2, M2.^2, dims=2)
                M4_cat = cat(M4, M4.^2, dims=2)

                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                            :component => KeyedArray(M2.^2; time=1:3, id=[:a, :b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                            :component => KeyedArray(M4.^2; time=1:3, id=[:a, :b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply_append(ds, p, (:_, :price, :_); append_dim=2)

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)

                expected = KeyedDataset(
                    flatten([
                        :train => [
                            :load => KeyedArray(M1; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M2; time=1:3, id=[:a, :b]),
                            :price2 => KeyedArray(M2.^2; time=1:3, id=[:a, :b]),
                        ],
                        :predict => [
                            :load => KeyedArray(M3; time=1:3, loc=[:x, :y]),
                            :price => KeyedArray(M4; time=1:3, id=[:a, :b]),
                            :price2 => KeyedArray(M4.^2; time=1:3, id=[:a, :b]),
                        ]
                    ])...
                )

                r = FeatureTransforms.apply_append(
                    ds, p, (:_, :price, :_);
                    component_name=:price2, append_dim=2
                )

                @test r isa KeyedDataset
                @test isequal(r, expected)
                @test !isequal(ds, expected)
            end
        end
    end
end
