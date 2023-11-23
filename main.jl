using cats_n_dogs
using Flux
using MLUtils
using Metalhead
using CUDA
using cuDNN
using ArgParse

device = CUDA.functional() ? gpu : cpu

function parse_cli(ARGS)
	settings = ArgParseSettings()
    @add_arg_table settings begin
        "--epochs"
            help = "Supply number of epochs to train model"
            arg_type = Int
            default = 5
    end
    return parse_args(ARGS, settings)
end

args = parse_cli(ARGS)

const EPOCHS = args["epochs"]

const dataset_x, dataset_y = createDataset("./src/data/train")
const training_set, validation_set = splitobs((dataset_x, dataset_y), at=0.7, shuffle=true)

const BATCHSIZE = 16

training_iterator = Flux.DataLoader(
    training_set,
    batchsize = BATCHSIZE,
    collate = true,
    parallel = true,
)

validation_iterator = Flux.DataLoader(
    validation_set,
    batchsize = BATCHSIZE,
    collate = true,
    parallel = true,
)

device == gpu ? val_iter = CUDA.CuIterator(validation_iterator) : nothing
device == gpu ? train_iter = CUDA.CuIterator(training_iterator) : nothing

model_base = Metalhead.ResNet(18, pretrain = true).layers[1:end-1] |> device
model_tune = Chain(AdaptiveMeanPool((1,1)), Flux.flatten, Dense(512 => 3)) |> device

opt = Flux.setup(Flux.Optimisers.Adam(1e-3), model_tune)

for iter = 1:EPOCHS
    @time train_epoch!(model_base, model_tune; opt, train_iter)
    metric_train = evaluate(model_base, model_tune, train_iter)
    metric_eval = evaluate(model_base, model_tune, val_iter)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end

save_model(model_tune)
