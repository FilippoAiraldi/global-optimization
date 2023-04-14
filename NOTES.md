# Notes

## Visualization


```julia
testcase = TestCases.BraninHoo

# 2D contour plot
f = TestCases.braninhoo
n = 50
x = range(-1, 1, n)
y = range(-1, 1, n)
z = f.(x', y; scaled=true)
plt = contour(
    x,
    y,
    z,
    levels=[logrange(minimum(z), maximum(z), 15)...],
    color=cgrad(:imola, rev=false, scale=:exp),
    fill=true,
    fillalpha=0.33,
    legend=false,
    clabels=false,
    cbar=false,
    title="Branin-Hoo Function",
    xlabel="x",
    ylabel="y",
)
# or wireframe
# plt = plot(x, y, z, st=:wireframe)

display(plt)
```
