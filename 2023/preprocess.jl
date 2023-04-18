using PlotlyJS, Distributions, Statistics
using DataFrames, CSV
using LightGBM

function cor_heatmap(df::DataFrame)
    col_names = names(df)
    return heatmap(z=cor(Matrix(df)), x=col_names, y=col_names)
end

function distri(train::DataFrame, test::DataFrame; skip_cols=Symbol[])
    open("exercise/distributions.txt", "w") do f
        for col in names(train)
            if col ∉ skip_cols
                train_n = fit(Normal, train[!, col])
                train_μ, train_σ = params(train_n)
                test_n = fit(Normal, test[!, col])
                test_μ, test_σ = params(test_n)
                error_μ = round(abs(train_μ - test_μ) / abs(train_μ) * 100, digits=2)
                error_σ = round(abs(train_σ - test_σ) / train_σ * 100, digits=2)
                if error_μ > 10 || error_σ > 10
                    println(f, "# $col #")
                    println(f, "train = " * string(train_n))
                    println(f, "test  = " * string(test_n))
                    println(f, "μ error = $error_μ% , σ error = $error_σ%")
                    println(f, "----------------------------------------")
                end
            end
        end
    end
    return nothing
end

function get_importance(estimator::LightGBM.LGBMClassification,
    x_train::DataFrame, y_train::Vector,
    x_test::DataFrame, y_test::Vector)
    fit!(estimator, Matrix(x_train), y_train, (Matrix(x_test), y_test), verbosity=-1)
    imp = LightGBM.gain_importance(estimator)
    df_imp = DataFrame(name=names(x_train), importance=imp)
    sort!(df_imp, :importance, rev=true)
    open("exercise/importance.csv", "w") do io
        CSV.write(io, df_imp)
    end
    return nothing
end


