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
        for file in readdir("$path/$label")
            img = load("$path/$label/$file")
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

end # module cats_n_dogs
