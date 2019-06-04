#include "pch.h"
#include <iostream>

using namespace std;

int main()
{
	ios_base::sync_with_stdio(false);
	double mean, weight1, weight2, percentage;
	cout << "Enter the average mass of the isotopes and the mass of each of them." << endl;
	cin >> mean >> weight1 >> weight2;
	percentage = (100 * mean - 100 * weight1) / (weight2-weight1);
	printf("Percentage of the first isotope is %.3g", 100 - percentage);
	printf("%% and percentage of the second isotope is %.3g", percentage);
	printf("%%.");
	return 0;
}