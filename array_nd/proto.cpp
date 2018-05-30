#include <array>
#include <vector>
#include <iostream>

template<int N>
inline std::array<int, N> calc_strides(const std::array<int, N>& dims)
{
    std::array<int, N> strides;
    strides[0] = 1;
    for(int i=1; i<N; ++i)
        strides[i] = strides[i-1]*dims[i-1];

    return strides;
}

template<int N>
int dot(const std::array<int, N>& left, const std::array<int, N>& right)
{
    int sum = 0;
    for (int i=0; i<N; ++i)
        sum += left[i]*right[i];

    return sum;
}

template<int N>
int product(const std::array<int, N>& array)
{
    int product = array[0];
    for (int i=1; i<N; ++i)
        product *= array[i];

    return product;
}

template<int N>
struct Array
{
    Array(std::array<int, N> dims) :
        dims(dims),
        ncells(product<N>(dims)),
        data(ncells),
        strides(calc_strides<N>(dims))
    {
        for (int i=0; i<ncells; ++i)
            data[i] = i;
    }

    inline double& operator()(const std::array<int, N>& indices)
    {
        const int index = dot<N>(indices, strides);
        return data[index];
    }

    inline double operator()(const std::array<int, N>& index) const
    {
        const int i = dot<N>(index, strides);
        return data[i];
    }

    inline void operator()(const std::array<std::pair<int, int>, N>& ranges) const
    {
        std::cout << "a( ";
        for (int i=0; i<N; ++i)
            std::cout << ranges[i].first << ":" << ranges[i].second << ", ";
    }

    const std::array<int, N> dims;
    const int ncells;
    std::vector<double> data;
    const std::array<int, N> strides;
};

int main()
{
    Array<3> a({128, 96, 64});
    std::cout << a({127, 95, 63}) << std::endl;

    return 0;
}
