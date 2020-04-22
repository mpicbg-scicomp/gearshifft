#ifndef RESULT_ALL_HPP_
#define RESULT_ALL_HPP_

#include "result_benchmark.hpp"
#include "result_writer.hpp"
#include "traits.hpp"
#include "types.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>


namespace gearshifft {

  template<int T_NumberRuns,  // runs per benchmark including warmup
           int T_NumberWarmups,
           int T_NumberValues // recorded values per run
           >
  class ResultAll {

    friend class ResultWriter<T_NumberRuns, T_NumberWarmups, T_NumberValues>;

    using ResultBenchmarkT = ResultBenchmark<T_NumberRuns, T_NumberValues>;

  public:

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
                                        (lhs.getDimKind()==rhs.getDimKind() &&
                                         (
                                          lhs.getDim()<rhs.getDim() ||
                                          (lhs.getDim()==rhs.getDim() &&
                                           lhs.getExtentsTotal()<rhs.getExtentsTotal())
                                         ));
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
     * If verbosity flag is set, then std::cout receives result view.
     */
    void saveCSV(const std::string& fname,
                 const std::string& apptitle,
                 const std::string& meta_information,
                 double timerContextCreate,
                 double timerContextDestroy) {
      std::ofstream fs;
      const char sep=',';

      fs.open(fname, std::ofstream::out);
      fs.precision(11);
      fs << "; " << meta_information <<"\n"
         << "; \"Time_ContextCreate [ms]\", " << timerContextCreate << "\n"
         << "; \"Time_ContextDestroy [ms]\", " << timerContextDestroy  << "\n";
      // header
      fs << "\"library\",\"inplace\",\"complex\",\"precision\",\"dim\",\"kind\""
         << ",\"nx\",\"ny\",\"nz\",\"run\",\"id\",\"success\"";
      for(auto ival=0; ival<T_NumberValues; ++ival) {
        fs << sep << '"' << static_cast<RecordType>(ival) << '"';
      }
      fs << "\n";

      // data
      for(auto& result : results_) {
        std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
        std::string complex = result.isComplex() ? "Complex" : "Real";
        for(auto run=0; run<T_NumberRuns; ++run) {
          result.setRun(run);
          fs << "\"" << apptitle << "\"" << sep
             << "\"" << inplace  << "\"" << sep
             << "\"" << complex  << "\"" << sep
             << "\"" << result.getPrecision() << "\"" << sep
             << result.getDim() << sep
             << "\"" << result.getDimKindStr() << "\"" << sep
             << result.getExtents()[0] << sep
             << result.getExtents()[1] << sep
             << result.getExtents()[2] << sep
             << run << sep
             << result.getID();
          // was run successfull?
          if(result.hasError() && result.getErrorRun()<=run) {
            if(result.getErrorRun()==run)
              fs << sep << "\"" <<result.getError() << "\"";
            else
              fs << sep << "\"Skipped\""; // subsequent runs did not run
          } else {
            if(run<T_NumberWarmups)
              fs << sep << "\"" << "Warmup" << "\"";
            else
              fs << sep << "\"" << "Success" << "\"";
          }
          // measured time and size values
          for(auto ival=0; ival<T_NumberValues; ++ival) {
            fs << sep << result.getValue(ival);
          }
          fs << "\n";
        } // run
      } // result
      fs.close();
    } // write

    std::size_t size() {
      return results_.size();
    }

  private:
    std::vector< ResultBenchmarkT > results_;

  };

}
#endif
