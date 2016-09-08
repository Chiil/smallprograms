#include <vector>
#include <iostream>

class Tmp
{
    public:
        Tmp() : available(true), data(100) {}
        ~Tmp() { std::cout << "Tmp (available = " << is_available() << ") going out of scope..." << std::endl; }
        bool is_available() const { return available; }
        void lock()
        {
            std::cout << "Locking Tmp" << std::endl;
            available = false;
        }
        void release()
        {
            available = true;
            std::cout << "Unlocking Tmp" << std::endl;
        }
    private:
        bool available;
        std::vector<double> data;
};

class Tmp_container
{
    public:
        Tmp_container(const unsigned n) : tmp_vector(n) {}
        Tmp& get_tmp()
        {
            for (Tmp& tmp : tmp_vector)
                if (tmp.is_available())
                {
                    tmp.lock();
                    return tmp;
                }
            throw std::runtime_error("Out of tmp!");
        }
        bool is_idle() const
        {
            for (const Tmp& tmp : tmp_vector)
                if (!tmp.is_available())
                    return false;

            return true;
        }

    private:
        std::vector<Tmp> tmp_vector;
};

int main()
{
    try
    {
        Tmp_container tmp_container(2);

        Tmp& tmp1 = tmp_container.get_tmp();
        Tmp& tmp2 = tmp_container.get_tmp();
        tmp1.release();
        tmp2.release();
        // Tmp& tmp3 = tmp_container.get_tmp();

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
