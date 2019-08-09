#include <array>
#include <vector>
#include <algorithm>
#include <iostream>

template<int N, typename ...Args>
inline std::array<int, N> get_array(const Args&... args)
{
    static_assert(sizeof...(args) == N, "Dimension size and number of indices do not match");
    return std::array<int, N>{ args... };
}

template<int N>
inline std::array<int, N> calc_strides(const std::array<int, N>& dims)
{
    std::array<int, N> strides;
    strides[0] = 1;
    for (int i=1; i<N; ++i)
        strides[i] = strides[i-1]*dims[i-1];

    return strides;
}

template<int N>
inline int calc_index(
        const std::array<int, N>& indices,
        const std::array<int, N>& strides,
        const std::array<int, N>& offsets)
{
    int sum = 0;
    for (int i=0; i<N; ++i)
        sum += (indices[i]-offsets[i]-1)*strides[i];

    return sum;
}

template<int N>
inline std::array<int, N> calc_indices(
        int index, const std::array<int, N>& strides, const std::array<int, N>& offsets)
{
    std::array<int, N> indices;

    for (int i=N-1; i>=1; --i)
    {
        indices[i] = index / strides[i];
        index %= strides[i];
    }
    indices[0] = index;

    for (int i=0; i<N; ++i)
        indices[i] += offsets[i] + 1;

    return indices;
}

template<int N>
inline int product(const std::array<int, N>& array)
{
    int product = array[0];
    for (int i=1; i<N; ++i)
        product *= array[i];

    return product;
}

template<typename T, int N>
class Array
{
    public:
        template<typename ...Args>
        Array(Args... args) :
            dims(get_array<N>(args...)),
            ncells(product<N>(dims)),
            data(ncells),
            strides(calc_strides<N>(dims)),
            offsets({})
        {}

        template<typename ...Args>
        inline T& operator()(Args... indices)
        {
            const int index = calc_index<N>(get_array<N>(indices...), strides, offsets);
            return data[index];
        }

        template<typename ...Args>
        inline T operator()(Args... indices) const
        {
            const int index = calc_index<N>(get_array<N>(indices...), strides, offsets);
            return data[index];
        }

        /*
        inline Array<T, N> subset(
                const std::array<std::pair<int, int>, N> ranges) const
        {
            // Calculate the dimension sizes based on the range.
            std::array<int, N> subdims;
            std::array<bool, N> do_spread;

            for (int i=0; i<N; ++i)
            {
                subdims[i] = ranges[i].second - ranges[i].first + 1;
                // CvH how flexible / tolerant are we?
                do_spread[i] = (dims[i] == 1);
            }

            // Create the array and fill it with the subset.
            Array<T, N> a_sub(subdims);
            for (int i=0; i<a_sub.ncells; ++i)
            {
                std::array<int, N> index;
                int ic = i;
                for (int n=N-1; n>0; --n)
                {
                    index[n] = do_spread[n] ? 1 : ic / a_sub.strides[n] + ranges[n].first;
                    ic %= a_sub.strides[n];
                }
                index[0] = do_spread[0] ? 1 : ic + ranges[0].first;
                a_sub.data[i] = (*this)(index);
            }

            return a_sub;
        }
        */


    private:
        std::array<int, N> dims;
        int ncells;
        std::vector<T> data;
        std::array<int, N> strides;
        std::array<int, N> offsets;
};

int main()
{
    Array<double, 4> a(3, 1, 2, 3);
    double d = a(3, 1, 1, 3);

    std::cout << d << std::endl;

    return 0;
}
