#include <map>
#include <string>
#include <vector>

struct Field3d
{
    Field3d(int i) {}
    std::vector<double> data;
    std::vector<double> databot;
};

typedef std::map<std::string, Field3d> Field_map;

struct Fields
{
    Field_map at;
    Fields() { at.emplace("u", 10); }
    double* d (std::string s) { return at.at(s).data.data(); }
    double* db(std::string s) { return at.at(s).databot.data(); }
};

void kernel(double* a, double* abot) {}

void kernel_launcher(Fields& f)
{
    // Method 1
    kernel(f.at.at("u").data.data(), f.at.at("u").databot.data());

    // Method 2
    kernel(f.d("u"), f.db("u"));

    // Method 3
    double* u  = f.at.at("u").data.data();
    double* ub = f.at.at("u").databot.data();
    kernel(u, ub);
}

int main()
{
    Fields fields;
    kernel_launcher(fields);
}
