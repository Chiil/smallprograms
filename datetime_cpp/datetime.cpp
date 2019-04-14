#include <string>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>

std::tm calc_tm_actual(const std::tm& tm_start, const double time)
{
    std::tm tm_actual = tm_start;
    tm_actual.tm_sec += static_cast<int>(time);
    std::mktime(&tm_actual);
    return tm_actual;
}

double calc_day_of_year(const std::tm& tm_start, const double time)
{
    std::tm tm_actual = calc_tm_actual(tm_start, time);
    const double frac_day = ( tm_actual.tm_hour*3600.
                            + tm_actual.tm_min*60.
                            + tm_actual.tm_sec + std::fmod(time, 1.) ) / 86400.;
    return tm_actual.tm_yday+1. + frac_day; // Counting starts at 0 in std::tm, thus add 1.
}

double calc_hour_of_day(const std::tm& tm_start, const double time)
{
    std::tm tm_actual = calc_tm_actual(tm_start, time);
    const double frac_hour = ( tm_actual.tm_min*60
                             + tm_actual.tm_sec + std::fmod(time, 1.) ) / 3600.;
    return tm_actual.tm_hour + frac_hour; // Counting starts at 0 in std::tm, thus add 1.
}

std::string make_string(const std::tm& tm)
{
    std::stringstream ss;
    // Year is relative to 1900, month count starts at 0.
    ss << std::setfill('0') << std::setw(4) << tm.tm_year+1900 << "-";
    ss << std::setfill('0') << std::setw(2) << tm.tm_mon+1     << "-";
    ss << std::setfill('0') << std::setw(2) << tm.tm_mday      << " ";
    ss << std::setfill('0') << std::setw(2) << tm.tm_hour      << ":";
    ss << std::setfill('0') << std::setw(2) << tm.tm_min       << ":";
    ss << std::setfill('0') << std::setw(2) << tm.tm_sec;
    return ss.str();
}

int main()
{
    std::string datetime_string("1982-04-26 10:30:11");

    std::tm tm_start{};
    strptime(datetime_string.c_str(), "%Y-%m-%d %H:%M:%S", &tm_start);

    double time = 0.;
    std::cout << "Datetime: " << make_string(calc_tm_actual(tm_start, time)) << std::endl;
    std::cout << "Day of the year: " << std::setprecision(15) << calc_day_of_year(tm_start, time) << std::endl;
    std::cout << "Fractional hour: " << std::setprecision(15) << calc_hour_of_day(tm_start, time) << std::endl;

    time = 3.*86400 + 5.*3600 + 2701;
    std::cout << "Datetime: " << make_string(calc_tm_actual(tm_start, time)) << std::endl;
    std::cout << "Day of the year: " << std::setprecision(15) << calc_day_of_year(tm_start, time) << std::endl;
    std::cout << "Fractional hour: " << std::setprecision(15) << calc_hour_of_day(tm_start, time) << std::endl;
}
