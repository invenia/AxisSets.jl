
@testset "Impute" begin
    ds = KeyedDataset(
        flatten([
            :train => [
                :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                :load => KeyedArray([7.0 7.7; 8.0 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
            ],
            :predict => [
                :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                :load => KeyedArray([7.0 7.7; 8.1 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
            ]
        ])...
    );

    @testset "validate" begin
        @test_throws ThresholdError Impute.threshold(ds; limit=0.1)
        r = Impute.threshold(ds; limit=1.0)

        for (k, v) in pairs(ds.data)
            @test isequal(r[k], v)
        end
    end

    @testset "declaremissings" begin
        nonmissing = KeyedDataset(
           flatten([
               :train => [
                   :temp => KeyedArray(allowmissing([1.0 1.1; -9999.0 2.2; 3.0 3.3]); time=1:3, id=[:a, :b]),
                   :load => KeyedArray(allowmissing([7.0 7.7; 8.0 NaN; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                   :temp => KeyedArray(allowmissing([1.0 -9999.0; 2.0 2.2; 3.0 3.3]); time=1:3, id=[:a, :b]),
                   :load => KeyedArray(allowmissing([7.0 7.7; 8.1 NaN; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ]
            ])...
        );

        r = Impute.declaremissings(nonmissing; values=(NaN, -9999.0))
        for (k, v) in ds.data
            @test isequal(r[k], v)
        end
    end

    @testset "impute" begin
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray(allowmissing([1.0 1.1; 2.2 2.2; 3.0 3.3]); time=1:3, id=[:a, :b]),
                    :load => KeyedArray(allowmissing([7.0 7.7; 8.0 8.0; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray(allowmissing([1.0 1.0; 2.0 2.2; 3.0 3.3]); time=1:3, id=[:a, :b]),
                    :load => KeyedArray(allowmissing([7.0 7.7; 8.1 8.1; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ]
            ])...
        );

        r = Impute.substitute(ds; dims=:time)
        for (k, v) in expected.data
            @test isequal(r[k], v)
        end

        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray(allowmissing([7.0 7.7; 8.0 8.8; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray(allowmissing([7.0 7.7; 8.1 8.8; 9.0 9.9]); time=1:3, loc=[:x, :y]),
                ]
            ])...
        );

        r = Impute.substitute(ds; dims=:loc)
        for (k, v) in expected.data
            @test isequal(r[k], v)
        end
    end

    @testset "filter" begin
        # All time axis are constraint, so they must match and removing a row from 1
        # component will remove it from all
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([3.0 3.3]; time=3:3, id=[:a, :b]),
                    :load => KeyedArray([9.0 9.9]; time=3:3, loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray([3.0 3.3]; time=3:3, id=[:a, :b]),
                    :load => KeyedArray([9.0 9.9]; time=3:3, loc=[:x, :y]),
                ]
            ])...
        );

        r = Impute.filter(ds; dims=:time)
        for (k, v) in expected.data
            @test isequal(r[k], v)
        end

        # Only :load has a shared :loc axis, so we see that that :y location is dropped from
        # both train and predict.
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0 8.0 9.0]'; time=1:3, loc=[:x]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0 8.1 9.0]'; time=1:3, loc=[:x]),
                ]
            ])...
        );

        r = Impute.filter(ds; dims=:loc)
        for (k, v) in expected.data
            @test isequal(r[k], v)
        end
    end
end
