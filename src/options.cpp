#include "core/options.hpp"

#include <iostream>
#include <fstream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

using namespace gearshifft;
namespace po = boost::program_options;


OptionsDefault::OptionsDefault() {
  add_options()
    ("help,h", "Print help messages")
    ("extent,e", po::value<std::vector<std::string>>()->multitoken()->
     composing(), "Specific extent (eg. 1024x1024) [>=1 nr. of args possible]")
    ("file,f", po::value<std::vector<std::string>>()->multitoken()->
     composing(), "File with extents (row-wise csv) [>=1 nr. of args possible]")
    ("output,o", po::value<std::string>(&outputFile_)->default_value("result.csv"), "output csv file, will be overwritten!")
    ("add-tag,t", po::value<std::string>(&tag_)->default_value(""), "Add custom tag to header of output file")
    ("verbose,v", "Prints benchmark statistics")
    ("version,V", "Prints gearshifft version")
    ("device,d", po::value<std::string>(&device_)->default_value("gpu"), "Compute device = (gpu|cpu|acc|<ID>). If device is not supported by FFT lib, then it is ignored and default is used.")
    ("ndevices,n", po::value<size_t>(&ndevices_)->default_value(0), "Number of devices (0=all), if supported by FFT lib (e.g. clfft and fftw with n CPU threads).")
    ("list-devices,l", "List of available compute devices with IDs, if supported.")
    ("list-benchmarks,b", "Show registered benchmarks")
    ("run-benchmarks,r", po::value<std::string>(), "Run specific benchmarks (wildcards possible, e.g. ClFFT/float/*/Inplace_Real)")
    ;
}

void OptionsDefault::parseFile(const std::string& file) {
  std::ifstream f(file);
  std::string line;
  while(std::getline(f, line)) {
    if(boost::starts_with(line, "#")) // skip comment line
      continue;
    if(line.empty())
      continue;
    parseExtent(line);
  }
}

void OptionsDefault::parseExtent( const std::string& extent ) {
  std::vector<std::string> token;
  boost::split(token, extent, boost::is_any_of("x,"));
  if(token.size()==1) {
    Extents1D array = {std::stoull(token[0])};
    vector1D_.push_back( array );
  } else if(token.size()==2) {
    Extents2D array = {std::stoull(token[0]),
                       std::stoull(token[1])};
    vector2D_.push_back( array );
  } else {
    Extents3D array = {std::stoull(token[0]),
                       std::stoull(token[1]),
                       std::stoull(token[2])};
    vector3D_.push_back( array );
  }
}

/// processes command line arguments and apply the values to the variables
int OptionsDefault::parse(int argc, char *argv[]) {

  po::variables_map vm;
  try {
    po::parsed_options parsed
      = po::command_line_parser(argc, argv).options(desc_).allow_unregistered().run();
    po::store(parsed, vm);
    if( vm.count("version")  ) {
      version_ = true;
      return 1;
    }
    if( vm.count("help")  ) {
      help_ = true;
      return 1;
    }
    if( vm.count("file")  ) {
      auto files = vm["file"].as<std::vector<std::string> >();
      for( auto f : files ) {
        parseFile(f);
      }
    }
    if( vm.count("extent")  ) {
      auto extents = vm["extent"].as<std::vector<std::string> >();
      for( auto e : extents ) {
        parseExtent(e);
      }
    }
    if( vm.count("verbose")  ) {
      verbose_ = true;
    }else{
      verbose_ = false;
    }
    if( vm.count("list-devices")  ) {
      listDevices_ = true;
      return 1;
    }else{
      listDevices_ = false;
    }
    // no file and no extent given, use default config
    if( !vm.count("file") && !vm.count("extent") ) {
      if( std::ifstream(BOOST_STRINGIZE(GEARSHIFFT_INSTALL_CONFIG_FILE)).good() ) {
        parseFile(BOOST_STRINGIZE(GEARSHIFFT_INSTALL_CONFIG_FILE));
      }else if(std::ifstream("../share/gearshifft/extents.conf").good()){ // if not configs installed yet, use local path
        parseFile("../share/gearshifft/extents.conf");
      }else{ // if default config file is missing, then use these extents
        std::cerr << "Could not find 'extents.conf' so internal fallback is used." << std::endl;
        parseExtent("32");
        parseExtent("32x32");
        parseExtent("32x32x32");
      }
    }

    // use Boost Test environment variables
    setenv("BOOST_TEST_REPORT_LEVEL", "no", 0);
    if( vm.count("list-benchmarks") ){
      listBenchmarks_ = true;
      setenv("BOOST_TEST_LIST_CONTENT", "", 0);
    }
    if( vm.count("run-benchmarks") ){
      setenv("BOOST_TEST_RUN_FILTERS", vm["run-benchmarks"].as<std::string>().c_str(), 0);
    }

    po::notify(vm);
  }
  catch(po::error& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << desc_ << std::endl;
    return 2;
  }

  return 0;
}
