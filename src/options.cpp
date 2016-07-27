#include "options.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

using namespace gearshifft;

void Options::parseFile(const std::string& file) {
  std::ifstream f(file);
  std::string line;
  while(std::getline(f, line)) {
    if(boost::starts_with(line, "#")) // skip comment line
      continue;
    if(line.empty())
      continue;

    std::replace( line.begin(), line.end(), ',', 'x');
    parseExtent(line);
  }
}

void Options::parseExtent( const std::string& extent ) {
  std::vector<std::string> token;
  boost::split(token, extent, boost::is_any_of("x"));
  if(token.size()==1) {
    Extents1D array = {static_cast<unsigned>(std::stoi(token[0]))};
    vector1D_.push_back( array );
  } else if(token.size()==2) {
    Extents2D array = {static_cast<unsigned>(std::stoi(token[0])),
                       static_cast<unsigned>(std::stoi(token[1]))};
    vector2D_.push_back( array );
  } else {
    Extents3D array = {static_cast<unsigned>(std::stoi(token[0])),
                       static_cast<unsigned>(std::stoi(token[1])),
                       static_cast<unsigned>(std::stoi(token[2]))};
    vector3D_.push_back( array );
  }
}

/// processes command line arguments and apply the values to the variables
int Options::process(int argc, char* argv[]) {
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
    ("help,h", "Print help messages")
    ("extent,e", po::value<std::vector<std::string>>()->multitoken()->
     composing(), "specific extent (eg. 1024x1024) [>=1 nr. of args possible]")
    ("file,f", po::value<std::vector<std::string>>()->multitoken()->
     composing(), "file with extents (row-wise csv) [>=1 nr. of args possible]")
    ("output,o", po::value<std::string>(&outputFile_)->default_value("result.csv"), "output csv file location")
    ("verbose,v", "for console output");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc),
              vm); // can throw

    if( vm.count("help")  ) {
      std::cout << "gearshifft extra command line parameters" << std::endl
                << desc << std::endl;
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
    // no file and no extent given, use default config
    if( !vm.count("file") && !vm.count("extent") ) {
      if( std::ifstream("../config/extents.csv").good() ) {
        parseFile("../config/extents.csv");
      }else{ // if default config file is missing, then use these extents
        std::cerr << "Could not find '../config/extents.txt' so using default." << std::endl;
        parseExtent("32");
        parseExtent("32x32");
        parseExtent("32x32x32");
      }
    }
    po::notify(vm);
  }
  catch(po::error& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return -1;
  }
  return 0;
}
