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

    @testset "Reductions" begin
        @testset "Pattern(:__, :time)" begin
            reduced = Pattern(:__, :time)

            patterns = Pattern[
                (:__, :_, :time),
                (:_, :__, :time),
                (:__, :__, :time),
                (:_, :__, :_, :time),
            ]
            @testset "$p" for p in patterns
                @test p == reduced
            end

            @test Pattern(:_, :_, :time) != reduced
            @test Pattern(:_, :__, :time, :_) != reduced
            @test Pattern(:time, :_, :__) != reduced
        end

        @testset "Pattern(:train, :__)" begin
            reduced = Pattern(:train, :__)

            patterns = Pattern[
                (:train, :_, :__),
                (:train, :__, :_),
                (:train, :__, :__),
                (:train, :_, :__, :_),
            ]
            @testset "$p" for p in patterns
                @test p == reduced
            end

            @test Pattern(:train, :_, :_) != reduced
            @test Pattern(:_, :train, :__, :_) != reduced
            @test Pattern(:__, :_, :train) != reduced
        end

        @testset "Pattern(:_, :input, :__)" begin
            reduced = Pattern(:_, :input, :__)

            @test Pattern(:_, :input, :_, :__) == reduced
            @test Pattern(:_, :__, :input, :__) != reduced
        end
    end
end
