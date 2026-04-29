# Derivative TouchDesigner plugin SDK headers

The two headers in this directory — `TOP_CPlusPlusBase.h` and
`CPlusPlus_Common.h` — are property of Derivative Inc. and are redistributed
here under their "Shared Use License" (see the license header at the top of
each file).

## Source

These specific copies were taken from the SpectrumTOP sample in
[TouchDesigner CustomOperatorSamples](https://github.com/TouchDesigner/CustomOperatorSamples)
because that sample uses TOP API version 12, which is required for
`TOP_ExecuteMode::CUDA` to function correctly. Older copies of these headers
in the same repo (e.g. under `BasicFilterTOP/`) declare API version 11, which
TouchDesigner silently downgrades to CPU-only execution.

To upgrade these headers, replace both files with newer copies from a future
SpectrumTOP / CustomOperatorSamples release; do not modify them in place.

## License

Per the header in each file, redistribution is permitted as long as:

1. The Derivative copyright notice and license header are preserved.
2. Derivative's name and trademarks are not used to endorse derivative
   products without prior written permission.

The headers are licensed for use *only* in conjunction with TouchDesigner.
