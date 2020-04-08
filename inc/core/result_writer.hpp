#ifndef RESULT_WRITER_HPP_
#define RESULT_WRITER_HPP_

#include "result_benchmark.hpp"

#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>


namespace gearshifft {

  template<int T_NumberRuns, int T_NumberWarmups, int T_NumberValues>
  class ResultAll;

  template<int T_NumberRuns, int T_NumberWarmups, int T_NumberValues>
  class ResultWriter {

    using ResultBenchmarkT = ResultBenchmark<T_NumberRuns, T_NumberValues>;
    using ResultAllT = ResultAll<T_NumberRuns, T_NumberWarmups, T_NumberValues>;

  public:

    ResultWriter(ResultAllT &resultAll) : resultAll_(resultAll), t_(&ResultWriter::run, this) {}

    ~ResultWriter() {
      t_.join();
    }

    void notify() {
      notification_.notify_one();
    }

    void run() {
      while(!resultAll_.done_) {
        std::unique_lock<std::mutex> l(resultAll_.resultsMutex_);
        notification_.wait(l);

        auto size = resultAll_.results_.size();
        l.unlock();

        while (cursor_ < size) {
          std::cout << "Printing " << cursor_++ + 1 << "/" << size << std::endl;
        }

      }
    }

    void printHeader(std::ostream& stream,
                     const std::string& apptitle,
                     const std::string& dev_infos,
                     double timerContextCreate,
                     double timerContextDestroy) {
      std::stringstream ss;
      ss << "; " << dev_infos << "\n"
         << "; \"Time_ContextCreate [ms]\", " << timerContextCreate << "\n"
         << "; \"Time_ContextDestroy [ms]\", " << timerContextDestroy  << "\n";
      ss << apptitle
         << ", RunsPerBenchmark="<<T_NumberRuns
         << "\n";

      stream << ss.str() << std::endl;
    }

    void printResult(std::ostream& stream, ResultBenchmarkT& result) {

      int nruns = T_NumberRuns;
      std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
      std::string complex = result.isComplex() ? "Complex" : "Real";
      std::stringstream ss;

      ss << std::setfill('-') << std::setw(70) <<"-"<< "\n";
      ss << inplace
         << ", "<<complex
         << ", "<<result.getPrecision()
         << ", Dim="<<result.getDim()
         << ", Kind="<<result.getDimKindStr()<<" ("<<result.getDimKind()<<")"
         << ", Ext="<<result.getExtents()
         << "\n";
      if(result.hasError()) {
        ss << " Error at run="<<result.getErrorRun()
           << ": "<<result.getError()
           << "\n";
        nruns = result.getErrorRun()+1;
      }
      ss << std::setfill('-') << std::setw(70) <<"-"<< "\n";
      ss << std::setfill(' ');
      double sum;
      for(int ival=0; ival<T_NumberValues; ++ival) {
        sum = 0.0;
        for(int run=T_NumberWarmups; run<nruns; ++run) {
          result.setRun(run);
          sum += result.getValue(ival);
        }
        ss << std::setw(28)
          << static_cast<RecordType>(ival)
           << ": " << std::setw(16) << sum/(T_NumberRuns-T_NumberWarmups)
           << " [avg]"
           << "\n";
      }
      stream << ss.str() << std::endl;
    }

  private:
    ResultAllT& resultAll_;
    std::thread t_;
    std::condition_variable notification_;
    std::size_t cursor_ = 0;

  };

}

#endif
