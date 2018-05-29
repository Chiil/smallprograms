#include <array>
#include <vector>

template<int dim>
inline std::array<int, dim> calc_strides(const std::array<int, dim>& dims)
{
    std::array<int, dim> strides;
    strides[0] = 1;
    for(int i=1; i<dim; ++i)
        strides[i] = strides[i-1]*dims[i-1];

    return strides;
}

template<int dim>
int dot(const std::array<int, dim> left, const std::array<int, dim> right)
{
    int sum = 0;
    for (int i=0; i<dim; ++i)
        sum += left[i]*right[i];

    return sum;
}

template<int dim>
struct Array
{
    Array(std::array<int, dim> dims) :
        dims(dims), strides(calc_strides<dim>(dims))
    {}

    double& operator()(std::array<int, dim> indices)
    {
        const int index = dot(indices, strides);
        return data[index];
    }

    const std::array<int, dim> dims;
    const std::array<int, dim> strides;
    std::vector<double> data;
};

int main()
{
    Array<3> a({128, 96, 64});
}
