# Ensuring that I have the required Flux and all other necessary packages installed.
using Pkg
Pkg.add("Flux")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
Pkg.add("Random")
Pkg.add("Dates")
Pkg.add("Plots")
Pkg.add("Statistics")
Pkg.add("MLDatasets")
Pkg.add("ScikitLearn)

# Loading of the required packages
using Flux
using CSV
using DataFrames
using StatsBase
using Random
using Dates
using Statistics
using Plots
using MLDatasets
using SkikitLearn

# Reading and plotting the dataset
# Convesion to DateTime array
df = CSV.read("sales-of-shampoo-over-a-three-year.csv", DataFrame, header=1)
df[!,:Month] = map(x -> "20$x",df[!,:Month])
first(df[!,:Month], 36)

# Confirm the changes to the dataset
display(df)

# Sorting the dataset
sort!(df, :Month)

# Further adjustments to the date format. 
df = CSV.read("sales-of-shampoo-over-a-three-year.csv", DataFrame, dateformat="yyyy-uu-dd", types=[String, Float64])
    df.Month = Date.(df.Month, "yyyy-uu-dd")
df[!,:Month] = map(x -> "2$x",df[!,:Month])
first(df[!,:Month], 36)

#Confirm the new changes
display(df)

# Computing of the mean and standard deviation of the Sales column
using CSV, DataFrames, Plots, StatsBase

sales_mean = mean(df."Sales of shampoo over a three year period")
sales_sdt = std(df."Sales of shampoo over a three year period")

# Scaling the sales column using the z-score normalization
df.Sales_scaled = (df."Sales of shampoo over a three year period" .- sales_mean) ./ sales_sdt

#Confirmation of the output
display(df.Sales_scaled)

#Defining the training, validation and testing dataset
train_data = df[1:30, :]
test_data = df[31:end, :]
val_data = df[31:end, :]

# Building and defining the LSTM architechture
using Flux: Chain, LSTM, Dense, Dropout

input_dim = 1
hidden_dim = 32
output_dim = 1
batch_size = 1

model = Chain(
LSTM(input_dim, hidden_dim),
Dropout(0.2),
LSTM(hidden_dim, hidden_dim),
    Dropout(0.2),
    Dense(hidden_dim, output_dim),
    )

#Defining the loss function and the optimizer
using Flux: Losses

loss(x, y) = Flux.Losses.mse(x, y)


using Flux: @epochs

train_loader = Flux.Data.DataLoader(train_data, batch_size=batch_size, shuffle=false)

function train()
    @epochs 100 begin
        for(x,y) in train_loader
            x = reshape(x, 1, 1, size(x, 1))
            y = reshape(y, 1, size(y, 1))
            gs = gradient(() -> loss(x,y), Flux.params(model))
            Flux.update!(opt, Flux.params(model), gs)
        end
    end
end

train()

using Statistics

function evaluate()
    preds =[]
    for i in 1:size(val_data_scaled, 1)
        if i < hidden_dim
            push!(preds, val_data_scaled[i])
        else
            x = reshape(val_data_scaled[i-hidden_dim+1:1], 1, hidden_dim)
            y_pred = model(x)
            push!(preds, Flux.item(y_pred))
        end
    end
    rmse = sqrt(mean((preds - val_data_scaled)-^2))
    println("RMSE:", rmse)
end

evaluate()

using Dates

last_date = Dates.lastday(Date(df.Date[end], "y-m-d"))
forecast_dates = [Dates.format(Date(last_date + Month(i)), "y-m-d") for i in 1:12]

forecast_data_scaled = scaler.transform(convert(Matrix, forcast_data[:, 2]))

forecast_preds = []
for i in 1:12
    x= reshape(forecast_data_scaled[end-hidden_dim+:end], 1, 1, hidden_dim)
    y_pred = model(x)
    push!(forecast_preds, Flux.item


