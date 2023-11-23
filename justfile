run+EPOCHS:
	julia --project -t8 main.jl --epochs {{EPOCHS}}

activate:
	julia --project
