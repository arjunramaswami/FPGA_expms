# Measurement

- 19.4_hpc, 19.4_max
  - reuse host side allocated buffers with new and same data between iterations
  - new host side buffers every iteration
  - data size from 2^1 to 2^27
- deviceBuf
  - Same device buffer between iterations for both host side buffer cases
- memalign
  - mem align changed for new mem allocation with new data considering this is
    the problem scenario
- svm3DFFT
  - first and second phase split measurements for both host side buffer cases 
- svm1DFFT
  - 
