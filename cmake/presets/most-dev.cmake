# preset that turns on a wide range of packages, some of which require
# external libraries. Compared to all_on.cmake some more unusual packages
# are removed. The resulting binary should be able to run most inputs.

set(ALL_PACKAGES
  BODY
  BPM
  COLVARS
  COMPRESS
  EXTRA-COMMAND
  EXTRA-COMPUTE
  EXTRA-DUMP
  EXTRA-FIX
  EXTRA-MOLECULE
  EXTRA-PAIR
  GRAPHICS
  KSPACE
  LEPTON
  MANYBODY
  MEAM
  MISC
  ML-IAP
  ML-PACE
  ML-POD
  ML-SNAP
  ML-UF3
  MOFFF
  MOLECULE
  OPT
  RIGID)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()

set(BUILD_TOOLS ON CACHE BOOL "" FORCE)
