@testset "Patterns" begin
    items = [
        (:train, :input, :prices, :time),
        (:train, :input, :prices, :id),
        (:train, :input, :prices, :lag),
        (:train, :input, :load, :time),
        (:train, :input, :load, :id),
        (:train, :input, :temperature, :time),
        (:train, :input, :temperature, :id),
        (:train, :output, :prices, :time),
        (:train, :output, :prices, :id),
        (:predict, :input, :prices, :time),
        (:predict, :input, :prices, :id),
        (:predict, :input, :prices, :lag),
        (:predict, :input, :load, :time),
        (:predict, :input, :load, :id),
        (:predict, :input, :temperature, :time),
        (:predict, :input, :temperature, :id),
        (:predict, :output, :prices, :time),
        (:predict, :output, :prices, :id),
    ]

    @testset "Pattern(:__, :time)" begin
        pattern = Pattern(:__, :time)
        @test filter(in(pattern), items) == [
            (:train, :input, :prices, :time),
            (:train, :input, :load, :time),
            (:train, :input, :temperature, :time),
            (:train, :output, :prices, :time),
            (:predict, :input, :prices, :time),
            (:predict, :input, :load, :time),
            (:predict, :input, :temperature, :time),
            (:predict, :output, :prices, :time),
        ]
    end

    @testset "Pattern(:_, :_, :_, :time)" begin
        pattern = Pattern(:_, :_, :_, :time)
        @test filter(in(pattern), items) == [
            (:train, :input, :prices, :time),
            (:train, :input, :load, :time),
            (:train, :input, :temperature, :time),
            (:train, :output, :prices, :time),
            (:predict, :input, :prices, :time),
            (:predict, :input, :load, :time),
            (:predict, :input, :temperature, :time),
            (:predict, :output, :prices, :time),
        ]
    end

    @testset "Pattern(:_, :time)" begin
        pattern = Pattern(:_, :time)
        @test isempty(filter(in(pattern), items))
    end

    @testset "Pattern(:train, :input, :_, :time)" begin
        pattern = Pattern(:train, :input, :_, :time)
        @test filter(in(pattern), items) == [
            (:train, :input, :prices, :time),
            (:train, :input, :load, :time),
            (:train, :input, :temperature, :time),
        ]
    end

    @testset "Pattern(:__, :_, :time)" begin
        # Check that our Pattern is reduced in cases where we have extra wildcards.
        @test Pattern(:__, :_, :time) == Pattern(:__, :time)
    end

    @testset "Pattern(:_, :__, :time)" begin
        # Check that our Pattern is reduced in cases where we have extra wildcards.
        @test Pattern(:_, :__, :time) == Pattern(:__, :time)
    end

    @testset "Pattern(:train, :input, :__)" begin
        pattern = Pattern(:train, :input, :__)
        @test filter(in(pattern), items) == [
            (:train, :input, :prices, :time),
            (:train, :input, :prices, :id),
            (:train, :input, :prices, :lag),
            (:train, :input, :load, :time),
            (:train, :input, :load, :id),
            (:train, :input, :temperature, :time),
            (:train, :input, :temperature, :id),
        ]
    end
end
