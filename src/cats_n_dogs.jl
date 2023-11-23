module cats_n_dogs

export createDataset, evaluate, train_epoch!, save_model

using Flux
using Flux:update!
using DataAugmentation
using Metalhead
using Images
using JLD2
using Dates
using ArgParse

function createDataset(path)
    # pre-allocate arrays for "features" x and "targets" y 
    X = []
    y = []
    # this functioun supposes we have a top directory containing a sub directory each containing our images, named after the classes
    for label in readdir(path)
        for file in readdir("$path/$label")
            img = load("$path/$label/$file")
            im_size = (24, 24)
            # define a transformation which we will apply to all images
            transform = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
            # we will broadcast these arrays over our image data to normalize the values
            DATA_MEAN = [0.485f0, 0.456f0, 0.406f0]
            DATA_STD = [0.229f0, 0.224f0, 0.225f0]
            # grab the "itemdata" of the image and apply our transformation
            _data = itemdata(apply(transform, Image(img)))
            # collect data and cast to float32
            _data = collect(channelview(float32.(RGB.(_data))))
            # normalize with broadcasting operations
            data = permutedims((_data .- DATA_MEAN) ./ DATA_STD, (3, 2, 1))
            # push image data and corresponding label into x and y arrays
            push!(X, data)
            push!(y, label)
        end
    end
    return X,Flux.onehotbatch(y, ["cats", "dogs", "pandas"])
end

function train_epoch!(model_base, model_tune; opt, train_iter)
    for (x, y) in train_iter
        infer = model_base(x)
        gradients = gradient(model_tune) do z
            Flux.Losses.logitcrossentropy(z(infer), y)
        end
        update!(opt, model_tune, gradients[1])
    end
end

function evaluate(model_base, model_tune, val_iter)
    good = 0
    count = 0
    for (x, y) in val_iter
        good += sum(Flux.onecold(model_tune(model_base(x))) .== Flux.onecold(y))
        count += length(y)
    end
    acc = round(good / count, digits = 3)
    return acc
end

function save_model(trained_model)
    saved_model = Flux.state(trained_model)
    # time = string(now())
    jldsave("./models/mod_$(now())"; saved_model)
end

# function parse_cli()
# 	settings = ArgParseSettings()
#     @add_arg_table settings begin
#         "epochs"
#             help = "Supply number of epochs to train model"
#             arg_type = Int
#             default = 5
#     end
#     return parse_args(settings)
# end

end # module cats_n_dogs
