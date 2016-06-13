#ifndef RESULT_ALL_HPP_
#define RESULT_ALL_HPP_

#include "result_benchmark.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/test/unit_test.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

// compiler flags, set by cmake
#ifndef BENCH_PRECISION
 #define BENCH_PRECISION float
#endif
#define BENCH_PRECISION_STRING BOOST_PP_STRINGIZE(BENCH_PRECISION)
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
      /*
      static size_t index = 0;
      item.key = index++;
      item.fftlibrary = test_suite_name;
      item.fftkind = title;
      item.dimkind = getDimKind(extent);
      item.dim = extent.size();
      item.totalCount = 1;
      for(size_t i=0; i<item.dim; ++i) {
        item.extent[i] = extent[i];
        item.totalCount *= extent[i];
      }
      item.precision = BOOST_PP_STRINGIZE(BENCH_PRECISION);
      item.sizeDeviceDataBuffer = results.alloc_mem_in_bytes;
      item.sizeDevicePlanBuffer = results.plan_mem_in_bytes;
      item.statistic = results.stats;
      items.push_back(item);*/
    }
    /*
     * sort order:  fftkind -> dimkind -> dim -> nx*ny*nz -> run
     */
/*    void sortStatistics() {
      sort( items.begin( ), items.end( ), [ ]( const StatisticItem& lhs, const StatisticItem& rhs ) {
          return lhs.fftkind.compare(rhs.fftkind)!=0 ||
            lhs.dimkind<rhs.dimkind ||
                        (lhs.dimkind==rhs.dimkind &&
                         (lhs.dim<rhs.dim || lhs.dim==rhs.dim &&
                          (lhs.totalCount<rhs.totalCount || lhs.totalCount==rhs.totalCount &&
                           lhs.run<rhs.run
                          )
                         )
                        );
        });
    }*/
    ResultAll() = default;
    ~ResultAll() {
      write();
    }

    /**
     * Store results in csv file.
     * If verbosity flag 'v' is set, then std::cout receives result view.
     */
    void write() {
      // get test suite name
      std::string test_suite_name = (boost::unit_test::framework::get<boost::unit_test::test_suite>(boost::unit_test::framework::current_test_case().p_parent_id)).p_name;

      std::string fname = BOOST_PP_STRINGIZE(DEFAULT_RESULT_FILE)".csv";
      std::ofstream fs;
      char sep=',';
      fs.open(fname, std::ofstream::out);
      for(auto result : results_) {
        std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
        std::string complex = result.isComplex() ? "Complex" : "Real";
        for(auto run=0; run<T_NumberRuns; ++run) {
          result.setRun(run);
          fs << test_suite_name << sep
             << inplace << sep
             << complex << sep
             << BENCH_PRECISION_STRING << sep
             << result.getDim() << sep
             << result.getDimKind() << sep
             << result.getExtents()[0] << sep
             << result.getExtents()[1] << sep
             << result.getExtents()[2]
             << run << sep
             << result.getSizeDeviceDataBuffer() << sep
             << result.getSizeDevicePlanBuffer() << sep;
          for(auto ival=0; ival<T_NumberValues; ++ival) {
            fs << result.getValue(ival);
          }
          fs << std::endl;
        } // run
      } // result
      fs.close();

      // if verbosity flag then print results to std::cout
      int argc = boost::unit_test::framework::master_test_suite().argc;
      char** argv = boost::unit_test::framework::master_test_suite().argv;
      int verbose = 0;
      for(int k=0; k<argc; ++k) {
        if(strcmp(argv[k],"v")==0){
          verbose = 1;
          break;
        }
      }
      if(verbose)
      {
        std::stringstream ss;
        // header
        // ...
        for(auto result : results_) {
          std::string inplace = result.isInplace() ? "Inplace" : "Outplace";
          std::string complex = result.isComplex() ? "Complex" : "Real";
          ss << test_suite_name
             << inplace
             << complex
             << BENCH_PRECISION_STRING
             << result.getDim()
             << result.getDimKind()
             << result.getExtents()[0]
             << result.getExtents()[1]
             << result.getExtents()[2];
          for(auto run=0; run<T_NumberRuns; ++run) {
            result.setRun(run);
            ss << run
               << result.getSizeDeviceDataBuffer()
               << result.getSizeDevicePlanBuffer();
            for(auto ival=0; ival<T_NumberValues; ++ival) {
              ss << result.getValue(ival);
            }
          }
        }
        std::cout << ss.str();
/*
        int i_stats = results.stats.getLength()-1;
        // output on console
        std::cout << test_suite_name << " " << title << " "
                  << extents.size()
                  << "D";
        for(const auto& e : extents)
          std::cout << ", "<<e;
        std::cout << ", DevAllocs " << results.alloc_mem_in_bytes / 1048576.0
                  << '+' << results.plan_mem_in_bytes / 1048576.0
                  << " MiB, "
                  << results.stats.getLabel(i_stats) << " = "
                  << results.stats.getAverage(i_stats) << results.stats.getUnit(i_stats)
                  << std::endl;

        ss <<'\"'<< test_suite_name  << '\"'
           << ",\"" <<  title <<"_"<< extents.size() << "D" << '\"'
           << ",\"Precision\"," << BOOST_PP_STRINGIZE(BENCH_PRECISION)
           << ",\"AllocBuffer\"," <<  results.alloc_mem_in_bytes / 1048576.0 << ",\"MiB\""
           << ",\"AllocPlan\"," <<  results.plan_mem_in_bytes / 1048576.0 << ",\"MiB\""
           << ",\"Extent\"";
        for(const auto& e : extents)
          ss  << ',' << e;
        ss << std::endl;

        ss << results.stats << std::endl;*/
      }
    }

  private:
    std::vector< ResultBenchmarkT > results_;
  };

    /*
  template<int T_NumberRuns, int T_NumberValues>
  std::ostream& operator<<(
    std::ostream& os,
    const ResultAll<T_NumberRuns, T_NumberValues>& results)
  {
    const char sep=',';
    auto timedata = stats.items[0].statistics;
    // header
    os << "fftkind, precision, dimkind, dim, nx, ny, nz, run, AllocBuffer (bytes), AllocPlan (bytes)";
    // header:time-labels
    for (int i = 0; i < timedata.getLength(); ++i) {
      os << sep << '"' << timedata.getLabel(i) << '"';
    }
    // header:time-unit
    os << sep << '"' << timedata.getUnit(0) << '"' << std::endl;
    // data
    for(auto item : stats.items) {
      os << item.fftkind << sep
         << item.precision << sep
         << item.dimkind << sep
         << item.dim << sep
         << item.totalcount << sep;
      for(auto v : item.extent)
        os << v << sep;
      os << item.run << sep
         << item.sizeDeviceDataBuffer << sep
         << item.sizeDevicePlanBuffer << sep;
      timedata = item.statistics;
      // data:time
      for(int i=0;i<timedata.getLength(); ++i) {
        for(int j=0; j<timedata.getCount(i); ++j) {
          os << timedata.getValue(i,j) << sep;
        }
      }

    }
  }
    */
}
#endif
