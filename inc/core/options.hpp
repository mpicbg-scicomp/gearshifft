#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include "types.hpp"
#include <string>
#include <vector>

namespace gearshifft {

  /**
   * Extract and provide options given by command line arguments like extents,
   * input files and verbosity. Is a singleton.
   */
  class Options {

  public:

    static Options& getInstance() {
      static Options options;
      return options;
    }

    bool getVerbose() const {
      return verbose_;
    }

    bool getListDevices() const {
      return listDevices_;
    }

    const std::string& getOutputFile() const {
      return outputFile_;
    }

    const std::string& getDevice() const {
      return device_;
    }

    void parseFile(const std::string& file);

    void parseExtent( const std::string& extent );

    /// processes command line arguments and apply the values to the variables
    int parse(std::vector<char*>&, std::vector<char*>&);

    const Extents1DVec& getExtents1D() const {
      return vector1D_;
    }
    const Extents2DVec& getExtents2D() const {
      return vector2D_;
    }
    const Extents3DVec& getExtents3D() const {
      return vector3D_;
    }

  private:
    Options() = default;

  private:
    bool verbose_ = false;
    std::string outputFile_;
    std::string device_;
    bool listDevices_ = false;

    Extents1DVec vector1D_;
    Extents2DVec vector2D_;
    Extents3DVec vector3D_;
  };
}

#endif
