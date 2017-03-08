#ifndef CONTEXT_H
#define CONTEXT_H

#include "options.hpp"
#include <boost/core/noncopyable.hpp>
#include <string>
#include <type_traits>

namespace gearshifft {

  /**
   * Context base class for creating and destroying (FFT) context,
   whose methods are benchmarked.
   *
   * Program options are stored in static method options().
   * Context-specific attributes can be retrieved by the method context(),
   if class-template parameter is linked to the structure definition.
   * \tparam T_Options Program options coming from command-line parameters.
   * \tparam T_Attributes Optional class-type of context specific attributes.
   */
  template<typename T_Options = OptionsDefault, typename T_Attributes = void>
  struct ContextDefault : private boost::noncopyable{

    static const std::string title() {
      return "";
    }

    static std::string get_device_list() {
      return "";
    }

    std::string get_used_device_properties() {
      return "";
    }

    void create() {
    }

    void destroy() {
    }

    static T_Options& options() {
      static T_Options options;
      return options;
    }

    /// only enable method if T_Attributes is non-void
    template<typename T = T_Attributes>
    static
    typename std::enable_if<!std::is_same<T, void>::value, T>::type&
    context() {
      static T_Attributes attributes;
      return attributes;
    }

  };

}

#endif /* CONTEXT_H */
