#ifndef STATISTICS_H_
#define STATISTICS_H_

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

class Statistics
{
  public:
    Statistics();
    virtual
    ~Statistics();

    int add(const std::string& label, const std::string& unit="", bool invert=false, double factor=1.0);
    int append(const std::string& label, const std::string& unit="", bool invert=false, double factor=1.0);
    void setFactor(int index, double factor);
    void setFactorAll(double factor);
    void process(int index, double val);
    void process(double val);
    void init(int i);
    void resetAll();
    void toString() const;

    int getLength() const;
    const std::string& getLabel(int i) const;
    const std::string& getUnit(int i) const;
    int getCount(int i) const;
    double getMin(int i) const;
    double getMax(int i) const;
    double getAverage(int i) const;
    double getStdDeviation(int i) const;

  private:
    void check_index(int index) const;

  private:
    int _current_index;
    std::vector<std::string> _labels;
    std::vector<std::string> _units;
    std::vector<int> _count;
    std::vector<double> _factors;
    std::vector<bool> _inverts;
    std::vector<double> _min;
    std::vector<double> _max;
    std::vector<double> _sum;
    std::vector<double> _sumsq;
};

std::ostream& operator<<(std::ostream& os, const Statistics& stats);

//-----------------------------------------------------------------------------


inline int Statistics::getLength() const {
  return _labels.size();
}

inline const std::string& Statistics::getLabel(int i) const {
  check_index(i);
  return _labels[i];
}

inline const std::string& Statistics::getUnit(int i) const {
  check_index(i);
  return _units[i];
}

inline double Statistics::getMin(int i) const {
  check_index(i);
  return _min[i];
}
inline double Statistics::getMax(int i) const {
  check_index(i);
  return _max[i];
}

inline int
Statistics::getCount (int i) const
{
	check_index(i);
	return _count[i];
}

inline double Statistics::getAverage(int i) const {
  check_index(i);
  return _sum[i]/_count[i];
}

#endif /* STATISTICS_H_ */
