#include <vector>
#include <iostream>

struct Add 
{
    static inline double apply(const double left, const double right)
    {
        return left+right;
    }
};

struct Multiply
{
    static inline double apply(const double left, const double right)
    {
        return left*right;
    }
};

// OPERATOR NODE CLASS
// Operator node in expression tree.
template<class Left, class Op, class Right>
struct Operator
{
    Operator(const Left& left, const Right& right) : left(left), right(right) {}

    const Left& left;
    const Right& right;

    inline double operator()(const int i) const
    { 
        return Op::apply(left(i), right(i));
    }
};

// Template classes for the math operators.
template<class Left, class Right>
inline Operator<Left, Add, Right> operator+(const Left& left, const Right& right)
{
    return Operator<Left, Add, Right>(left, right);
}

template<class Left, class Right>
inline Operator<Left, Multiply, Right> operator*(const Left& left, const Right& right)
{
    return Operator<Left, Multiply, Right>(left, right);
}

class Array_1d
{
    public:
        Array_1d(const int itot) :
            itot(itot), data(itot)
        {
            std::fill(data.begin(), data.end(), 0.);
        }

        void print()
        {
            for (int i=0; i<itot; ++i)
                std::cout << i << " = " << (*this)(i) << std::endl;
        }

        double& operator()(const int i) { return data[i]; }
        double operator()(const int i) const { return data[i]; }

        template<class T>
        inline Array_1d& operator= (const T& __restrict__ expression)
        {
            #pragma clang loop vectorize(enable)
            #pragma GCC ivdep
            #pragma ivdep
            for (int i=0; i<itot; ++i)
                (*this)(i) = expression(i);

            return *this;
        }

        inline Array_1d& operator= (const double value)
        {
            #pragma clang loop vectorize(enable)
            #pragma GCC ivdep
            #pragma ivdep
            for (int i=0; i<itot; ++i)
                (*this)(i) = value;

            return *this;
        }
    private:
        const int itot;
        std::vector<double> data;
};

