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

    void start(ResultAllT *resultAll,
               const std::string& apptitle,
               const std::string& dev_infos,
               bool verbose) {
      resultAll_ = resultAll;
      verbose_   = verbose;

      if (verbose_) {
          printHeader(apptitle, dev_infos);
      }

      t_ = std::thread(&ResultWriter::loop, this);
    }

    void stop(double timeContextCreate, double timeContextDestroy) {
      updateInternal(resultAll_->results_.size(), true);

      t_.join();

      if (verbose_) {
        printFooter(timeContextCreate, timeContextDestroy);
      }
    }

    void update(std::size_t cursorEnd) {
      updateInternal(cursorEnd, false);
    }

  private:

    ResultAllT* resultAll_ = nullptr;
    bool stop_ = false;
    bool verbose_ = false;
    std::size_t cursor_ = 0;
    std::size_t cursorEnd_ = 0;
    std::ostream& verboseOutput_ = std::cout; // TODO: Make configurable
    std::thread t_;
    std::mutex cursorEndMutex_; // used for condition variable update_ and mutual exclusion on cursorEnd_ and stop_
    std::condition_variable update_;

    void updateInternal(std::size_t cursorEnd, bool stop) {
      std::unique_lock<std::mutex> l(cursorEndMutex_);
      stop_      = stop;
      cursorEnd_ = cursorEnd;
      l.unlock();     // unlock before notifying so that the woken up thread doesn't have to wait for mutex again
      update_.notify_one();
    }

    void loop() {
      bool stop = false;
      while (!stop) {
        std::unique_lock<std::mutex> l(cursorEndMutex_);
        update_.wait(l);
        stop = stop_;

        if (verbose_) {
          std::stringstream ss;
          while (cursor_ < cursorEnd_) {
            auto& result = resultAll_->results_[cursor_];

            int nruns = T_NumberRuns;
            std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
            std::string complex = result.isComplex() ? "Complex" : "Real";

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
              cursor_++;
            }
            verboseOutput_ << ss.str() << std::endl;
          }

        // TODO: Write to file

        l.unlock();
      }
    }

    void printHeader(const std::string& apptitle, const std::string& dev_infos) {
      std::stringstream ss;
      ss << "; " << dev_infos << "\n"
         << apptitle << ", RunsPerBenchmark=" << T_NumberRuns << "\n";

      verboseOutput_ << ss.str() << std::endl;
    }

    void printFooter(double timeContextCreate,
                     double timeContextDestroy) {
      std::stringstream ss;
      ss << "; \"Time_ContextCreate [ms]\", " << timeContextCreate << "\n"
         << "; \"Time_ContextDestroy [ms]\", " << timeContextDestroy  << "\n";

      verboseOutput_ << ss.str() << std::endl;
    }

  };

}

#endif
