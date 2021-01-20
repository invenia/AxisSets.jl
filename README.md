# AxisSets

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/AxisSets.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://invenia.github.io/AxisSets.jl/dev)
[![Build Status](https://github.com/invenia/AxisSets.jl/workflows/CI/badge.svg)](https://github.com/invenia/AxisSets.jl/actions)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


1. Container Dataset type which stores multiple `KeyedArray{T,N,<:NamedDimsArray}`s with shared axes.
2. Perform dimensional operations over multiple arrays while maintaining axis consistency

# Operations

1. Filter along a dimension across multiple arrays. For example, remove labelled features from `train_X` or `predict_X` if we're missing too much data in either.
2. Reorder values in multiple arrays based on re-ordering the shared axis values.
3. Apply any other axis mutations to all shared arrays at once.
4. Ability to easily merge and component arrays at once
5. Supports the tables API
