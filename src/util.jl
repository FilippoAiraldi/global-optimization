"""logrange(x1, x2, n)

Creates a range from `x1` to `x2` of `n` logarithmically-spaced steps.
"""
logrange(x1, x2, n) = (10^y for y in range(log10(x1), log10(x2), length=n))
