void c_multiply(double* array, double multiplier, int m, int n)
{
    for (int i=0; i<m; ++i)
        for (int j=0; j<n; ++j)
        {
            const int index = i*m + j;
            array[index] *= multiplier;
        }
}
