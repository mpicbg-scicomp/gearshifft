#ifndef RESULT_WRITER_HPP_
#define RESULT_WRITER_HPP_

#include "result_benchmark.hpp"

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
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
               std::string fname,
               std::string apptitle,
               std::string dev_infos,
               bool verbose) {
      resultAll_ = resultAll;
      fname_     = fname;
      fnameBak_  = fname + BAK_SUFFIX;
      apptitle_  = apptitle;
      dev_infos_ = dev_infos;
      verbose_   = verbose;

      if (verbose_) {
          headerOut();
      }

      std::ofstream fs(fnameBak_, std::ofstream::out);
      fs.precision(PREC);
      headerToStreamCSV(fs, PrintContextTimes::No); // Context times not known yet

      t_ = std::thread(&ResultWriter::loop, this);
    }

    void stop(double timeContextCreate,
              double timeContextDestroy) {
      timeContextCreate_  = timeContextCreate;
      timeContextDestroy_ = timeContextDestroy;

      updateInternal(StopLoop::Yes);
      t_.join();

      if (verbose_) {
        footerOut();
      }
      std::ofstream fs(fnameBak_, std::ofstream::app);
      footerToStreamCSV(fs);
      fs.close();

      resultAll_->sort();
      saveCSV();
      std::cout << "Results written to " << fname_ << std::endl;

      std::remove(fnameBak_.c_str()); // delete backup file
    }

    void update() {
      updateInternal(StopLoop::No);
    }

    void saveCSV() const {
      std::ofstream fs(fname_, std::ofstream::out);
      fs.precision(PREC);

      headerToStreamCSV(fs, PrintContextTimes::Yes);

      for(auto& result : resultAll_->results_) {
        resultToStreamCSV(fs, result);
      }
    }

  private:

    enum class PrintContextTimes : bool {
      Yes = true,
      No  = false
    };

    enum class StopLoop : bool {
      Yes = true,
      No  = false
    };

    static constexpr char BAK_SUFFIX[] = "~";
    static constexpr char SEP = ',';
    static constexpr std::streamsize PREC = 11;

    ResultAllT* resultAll_ = nullptr;
    StopLoop stopLoop_     = StopLoop::No;
    bool verbose_ = false;
    double timeContextCreate_  = 0.0;
    double timeContextDestroy_ = 0.0;
    std::size_t cursor_    = 0;
    std::size_t cursorEnd_ = 0;
    std::string fname_;
    std::string fnameBak_;
    std::string apptitle_;
    std::string dev_infos_;
    std::ostream& verboseOutput_ = std::cout; // TODO: Make configurable
    std::thread t_;
    std::mutex cursorEndMutex_; // used for condition variable update_ and mutual exclusion on cursorEnd_ and stop_
    std::condition_variable update_;

    void updateInternal(StopLoop stopLoop) {
      std::unique_lock<std::mutex> l(cursorEndMutex_);
      stopLoop_  = stopLoop;
      cursorEnd_ = resultAll_->size();
      l.unlock();     // unlock before notifying so that the woken up thread doesn't have to wait for mutex again
      update_.notify_one();
    }

    void loop() {

      auto stopLoop = StopLoop::No;
      while (stopLoop == StopLoop::No) {
        std::unique_lock<std::mutex> l(cursorEndMutex_);
        update_.wait(l);
        stopLoop = stopLoop_;

        std::stringstream ss;
        std::ofstream fs(fnameBak_, std::ostream::app);
        fs.precision(PREC);

        while (cursor_ < cursorEnd_) {
          auto& result = resultAll_->results_[cursor_];

          if (verbose_) {
            resultToStreamOut(ss, result);
          }
          resultToStreamCSV(fs, result);

          cursor_++;
        }

        if (verbose_) {
          verboseOutput_ << ss.str() << std::flush;
        }

        l.unlock();
      }
    }

    void headerOut() const {
      std::stringstream ss;
      ss << "; " << dev_infos_ << "\n"
         << apptitle_ << ", RunsPerBenchmark=" << T_NumberRuns << "\n";

      verboseOutput_ << ss.str() << std::flush;
    }

    void resultToStreamOut(std::stringstream& stream,
                              ResultBenchmarkT& result) const {
      int nruns = T_NumberRuns;
      std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
      std::string complex = result.isComplex() ? "Complex" : "Real";

      stream << std::setfill('-') << std::setw(70) <<"-"<< "\n"
             << inplace
             << ", "<<complex
             << ", "<<result.getPrecision()
             << ", Dim="<<result.getDim()
             << ", Kind="<<result.getDimKindStr()<<" ("<<result.getDimKind()<<")"
             << ", Ext="<<result.getExtents()
             << "\n";
      if(result.hasError()) {
        stream << " Error at run="<<result.getErrorRun()
               << ": "<<result.getError()
               << "\n";
        nruns = result.getErrorRun()+1;
      }
      stream << std::setfill('-') << std::setw(70) <<"-"<< "\n"
             << std::setfill(' ');
      double sum;
      for(int ival=0; ival<T_NumberValues; ++ival) {
        sum = 0.0;
        for(int run=T_NumberWarmups; run<nruns; ++run) {
          result.setRun(run);
          sum += result.getValue(ival);
        }
        stream << std::setw(28)
               << static_cast<RecordType>(ival)
               << ": " << std::setw(16) << sum/(T_NumberRuns-T_NumberWarmups)
               << " [avg]"
               << "\n";
      }
    }

    void footerOut() const {
      std::stringstream ss;
      ss << "; \"Time_ContextCreate [ms]\", " << timeContextCreate_ << "\n"
         << "; \"Time_ContextDestroy [ms]\", " << timeContextDestroy_  << "\n";

      verboseOutput_ << ss.str() << std::flush;
    }

    void headerToStreamCSV(std::ostream& stream,
                           PrintContextTimes printContextTimes) const {
      stream << "; " << dev_infos_ <<"\n";

      if (printContextTimes == PrintContextTimes::Yes) {
        stream << "; \"Time_ContextCreate [ms]\", " << timeContextCreate_ << "\n"
               << "; \"Time_ContextDestroy [ms]\", " << timeContextDestroy_  << "\n";
      }
      // header
      stream << "\"library\",\"inplace\",\"complex\",\"precision\",\"dim\",\"kind\""
             << ",\"nx\",\"ny\",\"nz\",\"run\",\"id\",\"success\"";
      for(auto ival=0; ival<T_NumberValues; ++ival) {
        stream << SEP << '"' << static_cast<RecordType>(ival) << '"';
      }
      stream << "\n";
    }

    void resultToStreamCSV(std::ostream& stream,
                           ResultBenchmarkT& result) const {
      std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
      std::string complex = result.isComplex() ? "Complex" : "Real";

      for(auto run=0; run<T_NumberRuns; ++run) {
        result.setRun(run);
        stream << "\"" << apptitle_ << "\"" << SEP
               << "\"" << inplace   << "\"" << SEP
               << "\"" << complex   << "\"" << SEP
               << "\"" << result.getPrecision() << "\"" << SEP
               << result.getDim() << SEP
               << "\"" << result.getDimKindStr() << "\"" << SEP
               << result.getExtents()[0] << SEP
               << result.getExtents()[1] << SEP
               << result.getExtents()[2] << SEP
               << run << SEP
               << result.getID();
        // was run successfull?
        if(result.hasError() && result.getErrorRun()<=run) {
          if(result.getErrorRun()==run)
            stream << SEP << "\"" <<result.getError() << "\"";
          else
            stream << SEP << "\"Skipped\""; // subsequent runs did not run
        } else {
          if(run<T_NumberWarmups)
            stream << SEP << "\"" << "Warmup" << "\"";
          else
            stream << SEP << "\"" << "Success" << "\"";
        }

        // measured time and size values
        for(auto ival=0; ival<T_NumberValues; ++ival) {
          stream << SEP << result.getValue(ival);
        }

        stream << "\n";
      }
    }

    void footerToStreamCSV(std::ostream& stream) {
      stream << "; \"Time_ContextCreate [ms]\", " << timeContextCreate_ << "\n"
             << "; \"Time_ContextDestroy [ms]\", " << timeContextDestroy_  << "\n";
    }

  };

  template<int T_NumberRuns, int T_NumberWarmups, int T_NumberValues>
  constexpr char ResultWriter<T_NumberRuns, T_NumberWarmups, T_NumberValues>::BAK_SUFFIX[];
}


#endif
