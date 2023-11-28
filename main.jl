using cats_n_dogs
using Flux
using MLUtils
using Metalhead
using CUDA
using cuDNN
using ArgParse

#set device variable as either gpu ot cpu depending on if a working cuda installation is present
device = CUDA.functional() ? gpu : cpu

#set an optional flag for number of epochs to train model, and parse what has been passed
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

#grab epochs variable from args
const EPOCHS = args["epochs"]

#create a dataset and load it into memory, then split it into training and validation sets at a 07/03 split
const dataset_x, dataset_y = createDataset("./src/data/train")
const training_set, validation_set = splitobs((dataset_x, dataset_y), at=0.7, shuffle=true)

#set batchzise to something fairly small as I'm running an older gpu locally
const BATCHSIZE = 16

#define variables for our two dataloaders. paralell makes it concurrent, collate puts all arrays in a batch into one array
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

#if cuda is available, make dataloaders CuIterators for increased efficiency
device == gpu ? val_iter = CUDA.CuIterator(validation_iterator) : nothing
device == gpu ? train_iter = CUDA.CuIterator(training_iterator) : nothing

# the base of our model is a pre-trained resnet model. omit the output layer and send to gpu or cpu
model_base = Metalhead.ResNet(18, pretrain = true).layers[1:end-1] |> device
#very simple model to fine-tune our results for current use case.
model_tune = Chain(AdaptiveMeanPool((1,1)), Flux.flatten, Dense(512 => 3)) |> device

#setup Adam optimizer
opt = Flux.setup(Flux.Optimisers.Adam(1e-3), model_tune)

#training loop
for iter = 1:EPOCHS
    @time train_epoch!(model_base, model_tune; opt, train_iter)
    metric_train = evaluate(model_base, model_tune, train_iter)
    metric_eval = evaluate(model_base, model_tune, val_iter)
    @info "train" metric = metric_train
    @info "eval" metric = metric_eval
end

save_model(model_tune)
