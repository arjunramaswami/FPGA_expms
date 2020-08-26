//  Author: Arjun Ramaswami

#ifndef HELPER_H
#define HELPER_H

#include <stdbool.h>
#include "bare.h"

bool create_data(float2 *inp, unsigned N);

void print_config(unsigned N, unsigned iter, bool interleaving);

void display_measures(double total_api_time, double pcie_rd, double pcie_wr, double exec, unsigned N, unsigned iter);

double getTimeinMilliseconds();
#endif // HELPER_H
