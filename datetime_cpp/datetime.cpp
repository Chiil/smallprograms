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

double calc_day_of_the_year(const std::tm& tm_start, const double time)
{
    std::tm tm_actual = calc_tm_actual(tm_start, time);
    const double frac_day = ( tm_actual.tm_hour*3600.
                            + tm_actual.tm_min*60.
                            + tm_actual.tm_sec + std::fmod(time, 1.) ) / 86400.;
    return tm_actual.tm_yday+1. + frac_day; // Counting starts at 0 in std::tm, thus add 1.
}

std::string make_string(const std::tm& tm)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << tm.tm_year  << "-";
    ss << std::setfill('0') << std::setw(2) << tm.tm_mon+1 << "-"; // Add one to let Jan == 1.
    ss << std::setfill('0') << std::setw(2) << tm.tm_mday  << " ";
    ss << std::setfill('0') << std::setw(2) << tm.tm_hour  << ":";
    ss << std::setfill('0') << std::setw(2) << tm.tm_min   << ":";
    ss << std::setfill('0') << std::setw(2) << tm.tm_sec;
    return ss.str();
}

int main()
{
    int year = 2000;
    int month = 12;
    int day = 31;
    int hour = 23;
    int minute = 30;
    int second = 0;

    std::tm tm_start{};
    tm_start.tm_year = year;
    tm_start.tm_mon  = month-1; // Months since Jan.
    tm_start.tm_mday = day;
    tm_start.tm_hour = hour;
    tm_start.tm_min  = minute;
    tm_start.tm_sec  = second;
    std::mktime(&tm_start);

    double time = 0.;
    std::cout << "Datetime: " << make_string(calc_tm_actual(tm_start, time)) << std::endl;
    std::cout << "Day of the year: " << std::setprecision(15) << calc_day_of_the_year(tm_start, time) << std::endl;

    time = 1801;
    std::cout << "Datetime: " << make_string(calc_tm_actual(tm_start, time)) << std::endl;
    std::cout << "Day of the year: " << std::setprecision(15) << calc_day_of_the_year(tm_start, time) << std::endl;
}
