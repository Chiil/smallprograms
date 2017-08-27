template<typename TF>
class Class_1
{
    TF value;
};

template<typename TF>
class Class_2
{
    TF value;
};

template<typename TF>
class Class_a
{
    public:
        Class_a(Class_1<TF>& c1, Class_2<TF>& c2) : c1(c1), c2(c2) {}
    private:
        Class_1<TF>& c1;
        Class_2<TF>& c2;
};

class Class_b
{
    public:
        template<typename TF> Class_b(Class_1<TF>& c1, Class_2<TF>& c2) : c1(c1), c2(c2) {}
    private:
        template<typename TF> Class_1<TF>& c1;
        template<typename TF> Class_2<TF>& c2;
};

int main()
{
    Class_1 cc1;
    Class_2 cc2;
    Class_a cca(cc1, cc2);
    Class_b ccb(cc1, cc2);

    return 0;
}
