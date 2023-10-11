module cats_n_dogs

using Flux
using Flux:update!
using DataAugmentation
using Metalhead
using Images

function createDataset(path)
    X = []
    y = []
    for label in readdir(path)
        for file in readdir("$path/$data/$label")
            img = load("$path/$data/$label/$file")
            im_size = (224, 224)
            transform = DataAugmentation.compose(ScaleKeepAspect(im_size), CenterCrop(im_size))
            #data = reshape(Float32.(channelview(img)),28,28,1)
            _data = itemdata(apply(transform, Image(img)))
            _data = collect(channelview(float32.(RGB.(_data))))
            data = permutedims(_data, (3, 2, 1))
            push!(X, data)
            push!(y, label)
        end
    end
    return X,Flux.onehotbatch(y, ["cats", "dogs"])
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

function train_epoch!(model_base, model_tune; opt, train_iter)
    for (x, y) in train_iter
        infer = model_base(x)
        gradients = gradient(model_tune) do z
            Flux.Losses.logitcrossentropy(z(infer), y)
        end
        update!(opt, model_tune, gradients[1])
    end
end

end # module cats_n_dogs
