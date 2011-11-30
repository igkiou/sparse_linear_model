#
# mexopts.sh	Shell script for configuring MEX-file creation script,
#               mex.  These options were tested with the specified compiler.
#
# usage:        Do not call this file directly; it is sourced by the
#               mex shell script.  Modify only if you don't like the
#               defaults after running mex.  No spaces are allowed
#               around the '=' in the variable assignment.
#
# Note: For the version of system compiler supported with this release,
#       refer to Technical Note 1601 at:
#       http://www.mathworks.com/support/tech-notes/1600/1601.html
#
#
# SELECTION_TAGs occur in template option files and are used by MATLAB
# tools, such as mex and mbuild, to determine the purpose of the contents
# of an option file. These tags are only interpreted when preceded by '#'
# and followed by ':'.
#
#SELECTION_TAG_MEX_OPT: Template Options file for building icc MEX-files
#
# Copyright 1984-2006 The MathWorks, Inc.
# $Revision: 1.78.4.13 $  $Date: 2007/03/10 16:10:46 $
#----------------------------------------------------------------------------
#
    TMW_ROOT="$MATLAB"
    MFLAGS=''
    if [ "$ENTRYPOINT" = "mexLibrary" ]; then
        MLIBS="-L$TMW_ROOT/bin/$Arch -lmx -lmex -lmat -lmwservices -lut"
    else  
        MLIBS="-L$TMW_ROOT/bin/$Arch -lmx -lmex -lmat"
    fi
    case "$Arch" in
        Undetermined)
#----------------------------------------------------------------------------
# Change this line if you need to specify the location of the MATLAB
# root directory.  The script needs to know where to find utility
# routines so that it can determine the architecture; therefore, this
# assignment needs to be done while the architecture is still
# undetermined.
#----------------------------------------------------------------------------
            MATLAB="$MATLAB"
            ;;
        glnx86)
#----------------------------------------------------------------------------
            RPATH="-Wl,-rpath-link,$TMW_ROOT/bin/$Arch"
            CC='icc'
            CFLAGS='-ansi -D_GNU_SOURCE -fexceptions'
            CFLAGS="$CFLAGS -fPIC -pthread -m32"
            CFLAGS="$CFLAGS -D_FILE_OFFSET_BITS=64" 
            CLIBS="$RPATH $MLIBS -lm -lstdc++"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#           
            CXX='icpc'
            CXXFLAGS='-ansi -D_GNU_SOURCE'
            CXXFLAGS="$CXXFLAGS -D_FILE_OFFSET_BITS=64" 
            CXXFLAGS="$CXXFLAGS -fPIC -pthread"
            CXXLIBS="$RPATH $MLIBS -lm"
            CXXOPTIMFLAGS='-O -DNDEBUG'
            CXXDEBUGFLAGS='-g'
#
#
#            FC='/opt/intel/composerxe-2011.1.107/bin/intel64/ifort'
			FC='gfortran'
            FFLAGS='-fexceptions'
            FFLAGS="$FFLAGS -fPIC"
            FLIBS="$RPATH $MLIBS -lm"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD="$COMPILER"
            
            LDEXTENSION='.mexglx'
            LDFLAGS="-pthread -shared -m32 -Wl,--version-script,$TMW_ROOT/extern/lib/$Arch/$MAPFILE -Wl,--no-undefined"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-g'
#
            POSTLINK_CMDS=':'
#----------------------------------------------------------------------------
            ;;
        glnxa64)
#----------------------------------------------------------------------------
# CC and CXX should be path to Intel's 64-bit compiler 
# note that cce is the 64-bit version and cc is the 32-bit version
            RPATH="-Wl,-rpath-link,$TMW_ROOT/bin/$Arch"
            CC='icc'
            CFLAGS='-DUSE_MATLAB_INTERFACE -D_GNU_SOURCE -fexceptions -DMKL_ILP64'
            CFLAGS="$CFLAGS -fPIC -fno-omit-frame-pointer -Wall -pthread"
            CFLAGS="$CFLAGS -I/opt/intel/mkl/10.2.5.035/include/ -I/opt/intel/Compiler/11.1/072/include/ -I/usr/local/cula/include/ -I/usr/local/cuda/include/"
            CLIBS="$RPATH $MLIBS"
            CLIBS="$CLIBS -L/opt/intel/Compiler/11.1/072/lib/intel64/"
            CLIBS="$CLIBS -L/opt/intel/mkl/10.2.5.035/lib/em64t/ -Wl,--start-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lsvml"
            CLIBS="$CLIBS -L/usr/local/cula/lib64/ -lcublas -lcudart -lcula"
            CLIBS="$CLIBS -lm"
            COPTIMFLAGS='-fast -no-prec-sqrt -DNDEBUG -parallel -openmp '
#			COPTIMFLAGS="$COPTIMFLAGS -fp-model fast=2"
#			COPTIMFLAGS="$COPTIMFLAGS -parallel"
#			COPTIMFLAGS="$COPTIMFLAGS -par-report3 -openmp-report2 -opt-report3 -vec-report3"
#			COPTIMFLAGS="$COPTIMFLAGS -guide"
#			COPTIMFLAGS="$COPTIMFLAGS -prof-gen"
#			COPTIMFLAGS="$COPTIMFLAGS -prof-use"
            CDEBUGFLAGS='-g'
#
            CXX='icpc'
            CXXFLAGS='-DUSE_MATLAB_INTERFACE -D_GNU_SOURCE -DMKL_ILP64'
            CXXFLAGS="$CXXFLAGS -fPIC -fno-omit-frame-pointer -Wall -pthread"
            CXXFLAGS="$CXXFLAGS -I/opt/intel/mkl/10.2.5.035/include/ -I/opt/intel/Compiler/11.1/072/include/ -I/usr/local/cula/include/ -I/usr/local/cuda/include/"
            CXXLIBS="$RPATH $MLIBS"
            CXXLIBS="$CXXLIBS -L/opt/intel/Compiler/11.1/072/lib/intel64/"
            CXXLIBS="$CXXLIBS -L/opt/intel/mkl/10.2.5.035/lib/em64t/ -Wl,--start-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lsvml"
            CXXLIBS="$CXXLIBS -L/usr/local/cula/lib64/ -lcublas -lcudart -lcula"
            CXXLIBS="$CXXLIBS -lm"
            CXXOPTIMFLAGS='-fast -no-prec-sqrt -DNDEBUG -parallel -openmp'
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -fp-model fast=2"
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -parallel"
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -par-report3 -openmp-report2 -opt-report 3 -vec-report3"
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -guide"
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -prof-gen"
#			CXXOPTIMFLAGS="$CXXOPTIMFLAGS -prof-use"
            CXXDEBUGFLAGS='-g'
#
#
#			FC='/opt/intel/composerxe-2011.1.107/bin/intel64/ifort'
            FC='gfortran'
            FFLAGS='-fexceptions'
            FFLAGS="$FFLAGS -fPIC -fno-omit-frame-pointer"
            FLIBS="$RPATH $MLIBS -lm"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD="$COMPILER"
            LDEXTENSION='.mexa64'
            LDFLAGS="-pthread -shared -Wl,--version-script,$TMW_ROOT/extern/lib/$Arch/$MAPFILE -Wl,--no-undefined -L/opt/intel/mkl/10.2.5.035/lib/em64t/libmkl_solver_ilp64.a -Wl,--start-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -openmp -lpthread -L/usr/local/cula/lib64/ -lcublas -lcudart -lcula"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-g'
#
            POSTLINK_CMDS=':'
#----------------------------------------------------------------------------
            ;;
        sol64)
#----------------------------------------------------------------------------
            CC='cc -xarch=v9a'
            CFLAGS='-dalign -xlibmieee -D__EXTENSIONS__ -D_POSIX_C_SOURCE=199506L -mt'
            CFLAGS="$CFLAGS -KPIC"
            CLIBS="$MLIBS -lm"
            CLIBS="$CLIBS -lc"
            COPTIMFLAGS='-xO3 -xlibmil -DNDEBUG'
            CDEBUGFLAGS='-xs -g'
#           
            CXX='CC -xarch=v9a -compat=5'
            CCV=`CC -xarch=v9a -V 2>&1`
            version=`expr "$CCV" : '.*\([0-9][0-9]*\)\.'`
            if [ "$version" = "4" ]; then
                    echo "SC5.0 or later C++ compiler is required"
            fi
            CXXFLAGS='-dalign -xlibmieee -D__EXTENSIONS__ -library=stlport4,Crun'
            CXXFLAGS="$CXXFLAGS -D_POSIX_C_SOURCE=199506L -mt"
            CXXFLAGS="$CXXFLAGS -KPIC -norunpath"
            CXXLIBS="$MLIBS -lm"
            CXXOPTIMFLAGS='-xO3 -xlibmil -DNDEBUG'
            CXXDEBUGFLAGS='-xs -g'
#
            FC='f90 -xarch=v9a'
            FFLAGS='-dalign'
            FFLAGS="$FFLAGS -KPIC -mt"
            FLIBS="$MLIBS -lfui -lfsu -lsunmath -lm -lc"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-xs -g'
#
            LD="$COMPILER"
            LDEXTENSION='.mexs64'
            LDFLAGS="-G -mt -M$TMW_ROOT/extern/lib/$Arch/$MAPFILE"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-xs -g'
#
            POSTLINK_CMDS=':'
#----------------------------------------------------------------------------
            ;;
        mac)
#----------------------------------------------------------------------------
            CC='gcc-4.0'
            CFLAGS='-fno-common -no-cpp-precomp -fexceptions'
            CLIBS="$MLIBS -lstdc++"
            COPTIMFLAGS='-O3 -fno-loop-optimize -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            CXX=g++-4.0
            CXXFLAGS='-fno-common -no-cpp-precomp -fexceptions -arch ppc'
            CXXLIBS="$MLIBS -lstdc++"
            CXXOPTIMFLAGS='-O3 -fno-loop-optimize -DNDEBUG'
            CXXDEBUGFLAGS='-g'
#
            FC='g95'
            FFLAGS='-fexceptions'
            FC_LIBDIR=`$FC -print-file-name=libf95.a 2>&1 | sed -n '1s/\/*libf95\.a//p'`
            FLIBS="$MLIBS -L$FC_LIBDIR -lf95"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD="$CC"
            LDEXTENSION='.mexmac'
            LDFLAGS='-Wl,-flat_namespace -undefined suppress'
            LDFLAGS="$LDFLAGS -bundle -Wl,-exported_symbols_list,$TMW_ROOT/extern/lib/$Arch/$MAPFILE"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-g'
#
            POSTLINK_CMDS=':'
#----------------------------------------------------------------------------
            ;;
        maci)
#----------------------------------------------------------------------------
            CC='gcc-4.0'
            CFLAGS='-fno-common -no-cpp-precomp -fexceptions'
            CLIBS="$MLIBS -lstdc++"
            COPTIMFLAGS='-O3 -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            CXX=g++-4.0
            CXXFLAGS='-fno-common -no-cpp-precomp -fexceptions -arch i386'
            CXXLIBS="$MLIBS -lstdc++"
            CXXOPTIMFLAGS='-O3 -DNDEBUG'
            CXXDEBUGFLAGS='-g'
#
            FC='g95'
            FFLAGS='-fexceptions'
            FC_LIBDIR=`$FC -print-file-name=libf95.a 2>&1 | sed -n '1s/\/*libf95\.a//p'`
            FLIBS="$MLIBS -L$FC_LIBDIR -lf95"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD="$CC"
            LDEXTENSION='.mexmaci'
            LDFLAGS='-Wl,-flat_namespace -undefined suppress'
            LDFLAGS="$LDFLAGS -bundle -Wl,-exported_symbols_list,$TMW_ROOT/extern/lib/$Arch/$MAPFILE"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-g'
#
            POSTLINK_CMDS=':'
#----------------------------------------------------------------------------
            ;;
    esac
#############################################################################
#
# Architecture independent lines:
#
#     Set and uncomment any lines which will apply to all architectures.
#
#----------------------------------------------------------------------------
#           CC="$CC"
#           CFLAGS="$CFLAGS"
#           COPTIMFLAGS="$COPTIMFLAGS"
#           CDEBUGFLAGS="$CDEBUGFLAGS"
#           CLIBS="$CLIBS"
#
#           FC="$FC"
#           FFLAGS="$FFLAGS"
#           FOPTIMFLAGS="$FOPTIMFLAGS"
#           FDEBUGFLAGS="$FDEBUGFLAGS"
#           FLIBS="$FLIBS"
#
#           LD="$LD"
#           LDFLAGS="$LDFLAGS"
#           LDOPTIMFLAGS="$LDOPTIMFLAGS"
#           LDDEBUGFLAGS="$LDDEBUGFLAGS"
#----------------------------------------------------------------------------
#############################################################################
