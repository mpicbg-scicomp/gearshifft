#ifndef RESULT_ALL_HPP_
#define RESULT_ALL_HPP_

#include "application.hpp"
#include "result_benchmark.hpp"
#include "traits.hpp"

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#ifndef DEFAULT_RESULT_FILE
 #define DEFAULT_RESULT_FILE results
#endif

namespace gearshifft {

  template<int T_NumberRuns,  // runs per benchmark
           int T_NumberValues // recorded values per run
           >
  class ResultAll {
  public:
    using ResultBenchmarkT = ResultBenchmark<T_NumberRuns, T_NumberValues>;

    void add(const ResultBenchmarkT& result) {
      results_.push_back(result);
    }
    /*
     * sort order:  fftkind -> dimkind -> dim -> nx*ny*nz
     */
    void sort() {
      std::stable_sort(
        results_.begin( ), results_.end( ),
        [ ]( const ResultBenchmarkT& lhs, const ResultBenchmarkT& rhs )
        {
          if(lhs.getPrecision()==rhs.getPrecision())
          if(lhs.isInplace()==rhs.isInplace())
            if(lhs.isComplex()==rhs.isComplex())
              return lhs.getDimKind()<rhs.getDimKind() ||
                     lhs.getDimKind()==rhs.getDimKind() &&
                     (
                       lhs.getDim()<rhs.getDim() ||
                       lhs.getDim()==rhs.getDim() &&
                       lhs.getExtentsTotal()<rhs.getExtentsTotal()
                     );
            else
              return lhs.isComplex();
          else
            return lhs.isInplace();
          else
            return false;
        });
    }

    /**
     * Store results in csv file.
     * If verbosity flag 'v' is set, then std::cout receives result view.
     */
    void write(const std::string& apptitle,
               const std::string& dev_infos,
               double timerContextCreate,
               double timerContextDestroy) {
      std::string fname = BOOST_PP_STRINGIZE(DEFAULT_RESULT_FILE)".csv";
      std::ofstream fs;
      const char sep=',';
      sort();
      fs.open(fname, std::ofstream::out);
      fs << "; " << dev_infos << std::endl
         << "; \"Time_ContextCreate [ms]\", " << timerContextCreate << std::endl
         << "; \"Time_ContextDestroy [ms]\", " << timerContextDestroy  << std::endl;
      // header
      fs << "\"library\",\"inplace\",\"complex\",\"precision\",\"dim\",\"kind\""
         << ",\"nx\",\"ny\",\"nz\",\"run\"";
      for(auto ival=0; ival<T_NumberValues; ++ival) {
        fs << sep << '"' << static_cast<RecordType>(ival) << '"';
      }
      fs << std::endl;

      // data
      for(auto result : results_) {
        std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
        std::string complex = result.isComplex() ? "Complex" : "Real";
        for(auto run=0; run<T_NumberRuns; ++run) {
          result.setRun(run);
          fs << apptitle << sep
             << inplace << sep
             << complex << sep
             << result.getPrecision() << sep
             << result.getDim() << sep
             << result.getDimKind() << sep
             << result.getExtents()[0] << sep
             << result.getExtents()[1] << sep
             << result.getExtents()[2] << sep
             << run;
          for(auto ival=0; ival<T_NumberValues; ++ival) {
            fs << sep << result.getValue(ival);
          }
          fs << std::endl;
        } // run
      } // result
      fs.close();

      // if verbosity flag then print results to std::cout
      if(verbose)
      {
        std::stringstream ss;

        ss << "; " << dev_infos << std::endl
           << "; \"Time_ContextCreate [ms]\", " << timerContextCreate << std::endl
           << "; \"Time_ContextDestroy [ms]\", " << timerContextDestroy  << std::endl;
        ss << apptitle
           << ", RunsPerBenchmark="<<T_NumberRuns
           << std::endl;
        for(auto result : results_) {
          std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
          std::string complex = result.isComplex() ? "Complex" : "Real";
          ss << inplace
             << ", "<<complex
             << ", "<<result.getPrecision()
             << ", Dim="<<result.getDim()
             << ", Kind="<<result.getDimKind()
             << ", Ext="<<result.getExtents()[0]
             << "x"<<result.getExtents()[1]
             << "x"<<result.getExtents()[2]
             << std::endl;
          double sum;
          for(auto ival=0; ival<T_NumberValues; ++ival) {
            sum = 0.0;
            for(auto run=0; run<T_NumberRuns; ++run) {
              result.setRun(run);
              sum += result.getValue(ival);
            }
            ss << " " << static_cast<RecordType>(ival)
               << ": ~" << sum/T_NumberRuns
               << std::endl;
          }
        }
        std::cout << ss.str() << std::endl;
      }
    } // write

  private:
    std::vector< ResultBenchmarkT > results_;
  };

}
#endif
