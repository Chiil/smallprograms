#include <vector>
#include <iostream>
#include <memory>

class Tmp
{
    public:
        Tmp() { std::cout << "Construct" << std::endl; }
        ~Tmp() { std::cout << "Destruct" << std::endl; }
};

class Tmp_container
{
    public:
        Tmp_container(int n)
        {
            for (int i=0; i<n; ++i)
                tmp_vector.emplace_back(std::make_unique<Tmp>());
        }
        std::unique_ptr<Tmp> get_tmp()
        {
            if (tmp_vector.empty())
                throw std::runtime_error("Out of tmp fields");

            std::unique_ptr<Tmp> tmp = std::move(tmp_vector.back());
            tmp_vector.pop_back();
            return tmp;
        }
        void release_tmp(std::unique_ptr<Tmp>& tmp)
        {
            tmp_vector.push_back(std::move(tmp));
        }
        void print_available() const
        {
            std::cout << "Available: " << tmp_vector.size() << std::endl;
        }

    private:
        std::vector<std::unique_ptr<Tmp>> tmp_vector;
};

int main()
{
    Tmp_container tmp_container(2);
    tmp_container.print_available();

    auto tmp1 = tmp_container.get_tmp();
    auto tmp2 = tmp_container.get_tmp();
    tmp_container.release_tmp(tmp2);

    tmp_container.print_available();

    std::cout << "Oink" << std::endl;

    return 0;
}

