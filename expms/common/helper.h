//  Author: Arjun Ramaswami

#ifndef HELPER_H
#define HELPER_H

#include <stdbool.h>
#include "bare.h"

bool fftf_create_data(float2 *inp, int N);

bool fft_create_data(double2 *inp, int N);

void print_config(unsigned N, unsigned iter);

void display_measures(double total_api_time, double pcie_rd, double pcie_wr, double exec, unsigned N, unsigned iter);

double getTimeinMilliseconds();
#endif // HELPER_H
