# ndspec

A library for working with ocean waves with a specific focus on energy density spectra.

## Summary

The library provides two main types (or Classes):
- `Spectrum1D` -- one-dimensional energy density spectra
- `Spectrum2D` -- two-dimensional energy density spectra

Each type provides various traits (methods) for calculating spectral
properties such as spectral moments and derived properties such as the
significant value, spectral periods, and the most probable maximum value
over a given time window. 

The library also provides various wave and wind models along with spreading
functions for constructing both the `Spectrum1D` and `Spectrum2D`
types.

## Installation

### Rust

From the command line:

```bash
cargo add ndspec
```

### Python

To install the python API:

```bash
pip install ndspec
```

## Documentation

See [https://docs.rs/ndspec/latest/ndspec/](https://docs.rs/ndspec/latest/ndspec/)

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md).










