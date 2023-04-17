# Setup Notes


Several libraries are required to configure and build this repo.

## Boost and CMake

 Boost.   Version 1.63 is needed for the boost/python/numpy.
          Version 1.58 is sufficient if you eliminate components
          that require numpy.

 CMake.   This configuraton is tested with CMake version 3.5

 It seems that for the FindPackage script in CMake
 to work without warnings, CMake version should ideally be 
 newer than Boost.   Technically Boost 1.63 requires at least
 CMake 3.7, but I have gotten it to work with only CMake 3.5

 https://stackoverflow.com/questions/42123509/cmake-finds-boost-but-the-imported-targets-not-available-for-boost-version

  On Fedora 36, I simply needed to install `boost-devel`.

## Python packages 

  On an Ubuntu system, I needed to install the following packages
  via `sudo apt-get install`

    - `python-imaging`   (for PIL)
    - `python-requests`  (for requests)
 
  Also, `tensorflow` was required to run the tests.
  One way to install tensorflow is via `pip`, e.g., `sudo pip install tensorflow`.


## Disclaimer

  Depending on the OS and Python versions you are using, package names
  might be different.  Mileage may vary.

  I tested this repo on the following configuration:

    - Ubuntu 22.04
    - python 3.6
    - Boost 1.76

