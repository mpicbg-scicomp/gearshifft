#ifndef OPTIONS_HPP_
#define OPTIONS_HPP_

#include "types.hpp"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <boost/program_options.hpp>
#include <boost/core/noncopyable.hpp>
#pragma GCC diagnostic pop
#include <string>

namespace gearshifft {

  /**
   * Extract and provide options given by command line arguments like extents,
   * input files and verbosity.
   *
   * By derivation further program options can be added.
   * See FftwOptions in libraries/fftw/fftw.hpp how to do this.
   */
  class OptionsDefault : private boost::noncopyable {

  public:

    OptionsDefault();

    bool getHelp() const {
      return help_;
    }

    bool getVerbose() const {
      return verbose_;
    }

    bool getVersion() const {
      return version_;
    }

    bool getListBenchmarks() const {
      return listBenchmarks_;
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

    const std::string& getTag() const {
      return tag_;
    }

    size_t getNumberDevices() const {
      return ndevices_;
    }

    auto add_options() {
      return desc_.add_options();
    }

    void parseFile(const std::string& file);

    void parseExtent( const std::string& extent );

    /// processes command line arguments and apply the values to the variables
    int parse(int, char *[]);

    const Extents1DVec& getExtents1D() const {
      return vector1D_;
    }
    const Extents2DVec& getExtents2D() const {
      return vector2D_;
    }
    const Extents3DVec& getExtents3D() const {
      return vector3D_;
    }

    const boost::program_options::options_description& getDescription() {
      return desc_;
    }

  protected:

    template<typename T>
    auto value(T* var) {
      return boost::program_options::value<T>(var);
    }

  private:

    std::string outputFile_;
    std::string device_;
    std::string tag_;

    size_t ndevices_ = 0;
    bool help_ = false;
    bool verbose_ = false;
    bool version_ = false;
    bool listBenchmarks_ = false;
    bool listDevices_ = false;

    Extents1DVec vector1D_;
    Extents2DVec vector2D_;
    Extents3DVec vector3D_;

    boost::program_options::options_description desc_ =
      boost::program_options::options_description("gearshifft options and flags");
  };
}

#endif
