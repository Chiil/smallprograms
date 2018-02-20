#include <vector>
#include <iostream>
#include <memory>

class Tmp
{
    public:
        Tmp() : data(100) {}
        ~Tmp() {}

    private:
        std::vector<double> data;
};

class Tmp_container
{
    public:
        Tmp_container(const unsigned n)
        {
            for (int i=0; i<n; ++i)
            {
                tmp_vector.push_back( std::make_pair( Tmp_status::Available, std::make_shared<Tmp>() ) );
            }
        }

        std::shared_ptr<Tmp> get_tmp()
        {
            for (auto& a : tmp_vector)
                if (a.first == Tmp_status::Available)
                {
                    a.first = Tmp_status::Locked;
                    return a.second;
                }
            throw std::runtime_error("Out of tmp!");
        }

        void release(std::shared_ptr<Tmp>& ptr)
        {
            auto result = std::find_if(tmp_vector.begin(), tmp_vector.end(),
                    [&ptr](Tmp_pair& tp) { return (tp.first == Tmp_status::Locked) && (tp.second == ptr); });

            if (result != tmp_vector.end())
            {
                std::cout << "Releasing Tmp field" << std::endl;
                result->first = Tmp_status::Available;
            }
            else
                throw std::runtime_error("Releasing a non-locked or non-exisiting Tmp field");
        }

        bool is_idle() const
        {
            for (const auto& a: tmp_vector)
            {
                if (a.first != Tmp_status::Available)
                    return false;
            }

            return true;
        }


    private:
        enum class Tmp_status { Available, Locked };
        typedef std::pair<Tmp_status, std::shared_ptr<Tmp>> Tmp_pair;
        std::vector<Tmp_pair> tmp_vector;
};

int main()
{
    try
    {
        Tmp_container tmp_container(2);

        auto tmp1 = tmp_container.get_tmp();
        auto tmp2 = tmp_container.get_tmp();
        tmp_container.release(tmp1);
        tmp_container.release(tmp2);

        if (!tmp_container.is_idle())
            throw std::runtime_error("Not all Tmp fields have been released at exit!");
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
