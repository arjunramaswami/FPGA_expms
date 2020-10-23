# FPGA Experiments

## Non-blocking PCIe Transfer tests

There are two functions on the host code:

1. `nb_pcie_test`: explicit pipeline step wise sychronization
2. `nb_event_pcie_test`: event based synchronization

- Both use double buffers in separate banks to pipeline PCIe device to host and host to device transfers.
- An empty kernel can be synthesized to give the path cmd line parameter to not error.
- `-n` is the number of complex floats to be transferred

```bash
cmake ..
make
make empty_emu

CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./nb_event_pcietest -n 134217728 -c 3 -i 2 -p emu_empty/empty.aocx
```
