
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
        @test isequal(r, ds)
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
        @test isequal(r, ds)
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
        @test isequal(r, expected)

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
        @test isequal(r, expected)
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
        @test isequal(r, expected)

        # All time axis are constraint but only :load is checked. :temp is updated to match
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.0 1.1; 3.0 3.3]; time=[1, 3], id=[:a, :b]),
                    :load => KeyedArray([7.0 7.7; 9.0 9.9]; time=[1, 3], loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0 missing; 3.0 3.3]; time=[1, 3], id=[:a, :b]),
                    :load => KeyedArray([7.0 7.7; 9.0 9.9]; time=[1, 3], loc=[:x, :y]),
                ]
            ])...
        );
        r = Impute.filter(ds; dims=Pattern(:__, :load, :time))
        @test isequal(r, expected)

        r = Impute.filter(ds; dims=(:__, :load, :time))
        @test isequal(r, expected)

        # Pattern must end in a dimname
        @test_throws ArgumentError Impute.filter(ds; dims=Pattern(:__, :load, :_))

        # Only :load has a shared :loc axis, so we see that that :y location is dropped from
        # both train and predict.
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0, 8.0, 9.0][:, :]; time=1:3, loc=[:x]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0, 8.1, 9.0][:, :]; time=1:3, loc=[:x]),
                ]
            ])...
        );

        r = Impute.filter(ds; dims=:loc)
        @test isequal(r, expected)

        constrained_ds = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.0 1.1; missing 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0 7.7; 8.0 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0 missing; 2.0 2.2; 3.0 3.3]; time=1:3, id=[:a, :b]),
                    :load => KeyedArray([7.0 7.7; 8.1 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ]
            ])...,
            constraints = Pattern[(:__, :time), (:__, :loc), (:train, :__, :id)]
        );

        # :id for :train is included in the constraints and :predict is unconstrained
        # Both drop only the :ids missing in their own table
        expected = KeyedDataset(
            flatten([
                :train => [
                    :temp => KeyedArray([1.1; 2.2; 3.3][:, :]; time=1:3, id=[:b]),
                    :load => KeyedArray([7.0 7.7; 8.0 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ],
                :predict => [
                    :temp => KeyedArray([1.0; 2.0; 3.0][:, :]; time=1:3, id=[:a]),
                    :load => KeyedArray([7.0 7.7; 8.1 missing; 9.0 9.9]; time=1:3, loc=[:x, :y]),
                ]
            ])...,
            constraints = Pattern[(:__, :time), (:__, :loc),  (:train, :__, :id)]
        );

        r = Impute.filter(constrained_ds; dims=:id)
        @test isequal(r, expected)


    end
end
