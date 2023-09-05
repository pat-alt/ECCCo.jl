function pre_process(x; noise::Float32=0.03f0)
    ϵ = Float32.(randn(size(x)) * noise)
    x += ϵ
    return x
end