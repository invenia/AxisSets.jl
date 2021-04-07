@testset "FeatureTransforms" begin
    ds = KeyedDataset(
        flatten([
            :train => [
                :load => KeyedArray([7.0 7.7; 8.0 8.2; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                :price => KeyedArray([-2.0 4.0; 3.0 2.0; -1.0 -1.0]; time=1:3, id=[:a, :b]),
            ],
            :predict => [
                :load => KeyedArray([7.0 7.7; 8.1 7.9; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                :price => KeyedArray([0.5 -1.0; -5.0 -2.0; 0.0 1.0]; time=1:3, id=[:a, :b]),
            ]
        ])...
    )

    @testset "transform" begin
        @test is_transformable(ds)
    end

    @testset "apply" begin
        p = Power(2)

        expected = KeyedDataset(
            flatten([
                :train => [
                    :price => KeyedArray([4.0 16.0; 9.0 4.0; 1.0 1.0]; time=1:3, id=[:a, :b]),
                ],
                :predict => [
                    :price => KeyedArray([0.25 1.0; 25.0 4.0; 0.0 1.0]; time=1:3, id=[:a, :b]),
                ]
            ])...
        )

        r = FeatureTransforms.apply(ds, p; dims=(:_, :price, :_))

        @test r isa KeyedDataset
        @test isequal(r, expected)

        expected = KeyedDataset(
            flatten([
                :train => [
                    :price => KeyedArray(hcat([16.0; 4.0; 1.0]); time=1:3, id=[:b]),
                ],
                :predict => [
                    :price => KeyedArray(hcat([1.0; 4.0; 1.0]); time=1:3, id=[:b]),
                ]
            ])...
        )

        r = FeatureTransforms.apply(ds, p; dims=(:_, :price, :id), inds=[2])

        @test r isa KeyedDataset
        @test isequal(r, expected)
    end
end
