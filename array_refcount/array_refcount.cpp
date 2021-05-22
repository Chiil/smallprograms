#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>


template<typename T>
class Array
{
    public:
        Array() { vec = std::make_shared<std::vector<T>>(); }

        // Copy and copy assignment is default.
        Array(const Array<T>& a) = default;
        Array<T>& operator=(const Array<T>& a) = default;

        // Move is followed by an assignment to leave an empty object.
        Array(Array<T>&& a)
        {
            this->vec = std::move(a.vec);
            a.vec = std::make_shared<std::vector<T>>();
        }

        Array<T>& operator=(Array<T>&& a)
        {
            this->vec = std::move(a.vec);
            a.vec = std::make_shared<std::vector<T>>();
        }

        T* data() const { return vec->data(); }
        T& operator[](const int idx) const { return vec->at(idx); }

        void resize(const int size)
        {
            if (vec->size() == 0)
                vec->resize(size);
            else
                throw std::runtime_error("Cannot resize non-zero array");
        }

        size_t size() const { return vec->size(); }

        Array<T> copy() const
        {
            Array<T> a_copy;
            *a_copy.vec = *this->vec;
            return a_copy;
        }

    private:
        std::shared_ptr<std::vector<T>> vec;
};


int main()
{
    Array<double> a;
    a.resize(10);
    a[3] = 666;

    Array<double> b(a);
    std::cout << a[3] << ", " << b[3] << std::endl;

    Array<double> c = a.copy();
    Array<double> d = c;

    c[3] = 777;
    std::cout << a[3] << ", " << b[3] << ", " << c[3] <<  ", " << d[3] << std::endl;

    Array<double> f(std::move(c));
    std::cout << c.size() << std::endl;

    return 0;
}
