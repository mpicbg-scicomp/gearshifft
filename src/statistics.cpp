#include "statistics.h"

#include <algorithm>
#include <stdio.h>
#include <limits>
#include <math.h>
#include <stdexcept>

using namespace std;
using namespace gearshifft::helper;

std::ostream& gearshifft::helper::operator<<(std::ostream& os, const Statistics& stats)
{
  const char sep = ',';
  os
     << setw(15) << "\"Label\"" << sep
     << setw(12) << "\"Min\"" << sep
     << setw(12) << "\"Avg\"" << sep
     << setw(12) << "\"Max\"" << sep
     << setw(12) << "\"Std\"" << sep
     << setw(6) << "\"Counts\"" << sep
     << setw(5) << "\"Unit\""
     << std::endl;
  for (int i = 0; i < stats.getLength(); ++i)
  {
    std::string label = std::string("\"")+stats.getLabel(i)+"\"";
    os
       << setw(15)
       << label << sep
       << setw(12)
       << stats.getMin(i) << sep
       << setw(12)
       << stats.getAverage(i) << sep
       << setw(12)
       << stats.getMax(i) << sep
       << setw(12)
       << stats.getStdDeviation(i) << sep
       << setw(6)
       << stats.getCount(i) << sep
       << setw(5)
       << stats.getUnit(i) << std::endl;
  }
  return os;
}


Statistics::Statistics()
{
  resetAll();
}

Statistics::~Statistics()
{
}

void Statistics::resetAll() {
  _labels.clear();
  _min.clear();
  _max.clear();
  _sum.clear();
  _sumsq.clear();
  _count.clear();
}

void Statistics::init(int i)
{
  int len = _labels.size();
  _min.resize(len);
  _max.resize(len);
  _sum.resize(len);
  _sumsq.resize(len);
  _units.resize(len);
  _factors.resize(len);
  _inverts.resize(len);
  _count.resize(len);
  _min[i] = std::numeric_limits<double>::max();
  _max[i] = std::numeric_limits<double>::min();
  _sum[i] = 0.0;
  _sumsq[i] = 0.0;
  _count[i] = 0;
}

int Statistics::add(const std::string& label, const std::string& unit, bool invert, double factor) {
  append(label, unit, invert, factor);
  init(_current_index);
  return _current_index;
}

int Statistics::append(const std::string& label, const std::string& unit, bool invert, double factor) {
  int i = -1;
  if(_labels.size()>0){
    i = std::find(_labels.begin(),_labels.end(),label) - _labels.begin();
    if(i>=static_cast<int>(_labels.size()))
      i=-1;
  }
  if(i==-1){
    _labels.push_back(label);
    _units.push_back(unit);
    i = _labels.size()-1;
    init(i);
  }
  _current_index = i;
  _factors[_current_index] = factor;
  _inverts[_current_index] = invert;
  return _current_index;
}

void
Statistics::setFactor(int index, double factor){
  check_index(index);
  _factors[index] = factor;
}

void
Statistics::setFactorAll(double factor)
{
  std::fill(_factors.begin(), _factors.end(), factor);
}

void
Statistics::process(double val)
{
  process(_current_index, val);
}

void
Statistics::process(int index, double val)
{
  check_index(index);
  _current_index = index;

  if(_inverts[_current_index])
    val = 1.0/val;

  val *= _factors[_current_index];

  if(_min[_current_index] > val){
    _min[_current_index] = val;
  }
  if(_max[_current_index] < val){
    _max[_current_index] = val;
  }
  _sum[_current_index] += val;
  _sumsq[_current_index] += val*val;
  ++_count[_current_index];
}
double Statistics::getStdDeviation(int i) const {
  check_index(i);
  if(_count[i]<2)
    return 0.0;
  return sqrt(1.0/(_count[i]-1)*(_sumsq[i] - _sum[i]*_sum[i]/_count[i]));
}

void Statistics::check_index(int index) const {
  if(index<0 || index>=getLength()){
    fprintf(stderr,"Index out of range (%d/%d).\n", index, getLength());
    throw std::out_of_range("");
  }
}

//@todo use cout
void
Statistics::toString () const
{
	printf("%5s %10s, %10s, %10s, %10s, %10s, %s, %s\n","Runs", "Min","Max","Avg","Std","Std%","Info", "Unit");
	for(int i=0; i<getLength(); ++i){
	  if(getCount(i)>0){
	    printf("%5d %10.3lf, %10.3lf, %10.3lf, %10.3lf, %10.3lf, \"%s\", \"%s\" \n",
		    getCount(i),
				getMin(i), getMax(i), getAverage(i), getStdDeviation(i), 100.0*getStdDeviation(i)/getAverage(i),
				getLabel(i).c_str(), getUnit(i).c_str());
	  }
	}
}
