# Interleaved vs. Planar

This is a performance comparison between relatively simple implementations of separable 1D convolutions for interleaved format vs. planar format.

## Get the code

Clone this repo:

```sh
git clone <this repo> c:\path\to\repo
```

## Build

Use the standard cmake workflow:

```sh
# for an x64 release build on Visual Studio 2019
mkdir c:\path\to\build\dir
cmake -G "Visual Studio 16 2019" -A x64 -Bc:\path\to\build\dir -Sc:\path\to\repo
cmake --build c:\path\to\build\dir --config release
```

## Functional Test

Run the test application.  These are extremely simple tests to confirm that the functions in the convolution library actually perform the operations correctly.

```sh
c:\path\to\build\dir\test\Release\test_convolution.exe
```

## Performance test

Run the performance test application:

```sh
# View the help screen
c:\path\to\build\dir\src\Release\interleaved_vs_planar.exe

# Perform 3 iterations of each performance test, using a 3000x2000x4 input matrix
c:\path\to\build\dir\src\Release\interleaved_vs_planar.exe 2000 3000 4 3
```

## Inspect results

The performance test application outputs the data in CSV format, which can be imported into a spreadsheet application like Excel or just viewed in the terminal.

```sh
# example output
test,horizontal,transpose,vertical,total
interleaved3,0.0778528,0,0.965919,1.04377
planar3,0.0512368,0,0.692819,0.744055
interleaved7,0.140819,0,0.928772,1.06959
planar7,0.10321,0,0.676833,0.780043
planar7withTranspose,0.10662,0.611383,0.104051,0.822054
```

There are 5 tests.  Every test operates on the same input data.

The values for the `horizontal`, `transpose`, `vertical`, and `total` are in seconds.

Each test is repeated `I` times back-to-back, where `I` is the value provided on the command line.  The reported time contains the details for the iteration that consumed the least `total` time.

1. interleaved3: Interpret the data as interleaved, and perform per-channel 2D separable blur using a kernel size of 3
1. planar3: Interpret the data as planar, and perform per-channel 2D separable blur using a kernel size of 3
1. interleaved7: Interpret the data as interleaved, and perform per-channel 2D separable blur using a kernel size of 7
1. planar7: Interpret the data as planar, and perform per-channel 2D separable blur using a kernel size of 7
1. planar7withTranspose: Interpret the data as planar, and perform per-channel 2D separable blur using a kernel size of 7.  However, rather than performing horizontal and vertical convolution, perform horizontal convolution, transpose, horizontal convolution again, and transpose again.

The time for the first horizontal convolution is reported in the `horizontal` column.  The time for the vertical (or in the case of the `planar7withTranspose`, the second horizontal) convolution is reported in the `vertical` column.  The `transpose` column reports the total time for the 2 transposes in the `planar7withTranspose` test, and 0 otherwise.  The `total` column reports the sum of the `horizontal`, `transpose`, and `vertical` columns.