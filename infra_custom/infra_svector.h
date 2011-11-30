//=============================================================================
// File Name: infra_svector.h
// Written by: Ofer Dekel (oferd@cs.huji.ac.il)
//
// Distributed as part of the infra C++ library for linear algebra
// Copyright (C) 2004 Ofer Dekel
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//=============================================================================
#ifndef _OFERD_SVECTOR_H_
#define _OFERD_SVECTOR_H_

//*****************************************************************************
// Included Files
//*****************************************************************************
#include <stdio.h>
#include <iostream>
#include <vector>

#define NULL_VALUE 0.0

namespace infra {
//*****************************************************************************
/** Implements a single coordiante in a sparse vector of double numbers.
    @author Ofer Dekel (oferd@cs.huji.ac.il)
*/
struct sentry {
//-----------------------------------------------------------------------------
/** Default constructor
*/
  inline sentry();
//-----------------------------------------------------------------------------
/** Constructor
*/
  inline sentry(unsigned long index, const double& value);
//-----------------------------------------------------------------------------
/** Copy constructor
*/
  inline sentry(const sentry& other);
//-----------------------------------------------------------------------------
/** operator < with an integer. Compares the index of the sentry with the given
    integer.
    @param i The integer to compare to
*/
  inline bool operator < (unsigned long i) const;

//-----------------------------------------------------------------------------
/** operator == with an integer. Compares the index of the sentry with the
    given integer.
    @param i The integer to compare to
*/
  inline bool operator == (unsigned long i) const;

//-----------------------------------------------------------------------------
/** Equality operator
    @param other The sentry being compared
*/
  inline bool operator == (const sentry& other) const;

/*============================================================================
 sentry members
=============================================================================*/
  unsigned long index;
  double value;
};

//*****************************************************************************
/** Implements a sparse vector of double numbers as a list of index-value 
    pairs. All unspecified indices are assumed to equal 0. 
    @author Ofer Dekel (oferd@cs.huji.ac.il)
*/
class svector
{
/******************************************************************************
 svector Inner classes
******************************************************************************/
//=============================================================================
//  Class reference
//=============================================================================
  class reference
    {
      friend class svector;
    public:
      inline operator const double () const;
      inline const double operator = (const double& s);
      inline const double operator *= (const double& s);
      inline const double operator /= (const double& s);
      inline const double operator += (const double& s);
      inline const double operator -= (const double& s);
    protected:
      inline reference(svector& v, unsigned long i);
      svector& _sv;
      const unsigned long _index;
    };

//=============================================================================
// friend declaration
//=============================================================================
friend std::istream& operator>> (std::istream& is, svector& s);
 
/*============================================================================
  svector Function Declarations
=============================================================================*/
 public:
//-----------------------------------------------------------------------------
/** Constructs a svector with a specified size (0 by default).
    @param size The total vector size 
*/
  inline svector(unsigned long size=0);

//-----------------------------------------------------------------------------
/** Copy constructor
    @param other A reference to the vector being copied.
*/
  inline svector(const svector& other);

//-----------------------------------------------------------------------------
/** Destructor
 */
  inline ~svector();
  
//-----------------------------------------------------------------------------
/** Swaps the memory and dimensions of this vector with the memory and
    dimensions of another vector.
    @param other The other vector
*/
  inline void swap(svector& other);

//-----------------------------------------------------------------------------
/** Changes the dimension of the svector. That is, the number of entries
    in the vector remains the same, only its (virtual) size is modified.
    If the new size is smaller than the previous size, any entry which
    exceeds the new dimension is discarded.
    @param size The new vector size.
*/
  inline void resize(unsigned long size);
  
//=============================================================================
// Binary file interface
//=============================================================================
//-----------------------------------------------------------------------------
/** Load a sparse vector from a binary file
    @param stream The stream that the vector is to be read from. 
*/
  inline svector(FILE* stream);

//-----------------------------------------------------------------------------
/** Load the svector from a binary file
    @param stream The stream from which the svector is read.
    @return A reference to this svector
*/
  inline svector& load_binary(FILE* stream);

//-----------------------------------------------------------------------------
/** Save the vector to a binary file
    @param stream The stream to which the vector will be written.
*/
  inline void save_binary(FILE* stream) const;

//=============================================================================
// Other initialization methods
//=============================================================================
//-----------------------------------------------------------------------------
/** Assignment operator
    @param other The svector being assigned
    @return A reference to this vector
*/
  inline svector& operator = (const svector& other);
  
//=============================================================================
// Access to basic parameters
//=============================================================================
//-----------------------------------------------------------------------------
/** Returns the vector's (virtual) size 
    @return The vector's size.
*/
  inline unsigned long size() const; 
  
//-----------------------------------------------------------------------------
/** Returns the number of non-null entries in the vector
    @return The number of non-null entries in the vector
*/
  inline unsigned long entries() const; 
  
//=============================================================================
// Methods of accessing and modifying the svector
//=============================================================================
//-----------------------------------------------------------------------------
/** Checks if a given entry in the vector is null (empty)
    @param index The index of the entry being queried
    @param index The index of the entry being referenced.
*/
  inline bool is_null(unsigned long index) const;
  
//-----------------------------------------------------------------------------
/** Sets an entry in the vector to equal a non-zero value. If the entry was
    already non-null, its value is updated. If it was null, it is allocated
    and then set.
    @param index The index of the entry being referenced
    @param value The value being set
    @return A reference to this svector
*/
  inline svector& set(unsigned long index, const double& value);

//-----------------------------------------------------------------------------
/** Gets a given coordiante in the svector. If this coordiante is null, then
    a null is returned.
    @param index The index of the entry being referenced
*/
  inline const double get(unsigned long index) const;

//-----------------------------------------------------------------------------
/** Sets a non-zero entry in the vector to equal zero, in fact erasing it
    completely from memory. If erase is called on an entry which is already
    null, nothing happens.
    @param index The index of the entry being erased
    @return A reference to this svector
*/
  inline svector& erase(unsigned long index);

//-----------------------------------------------------------------------------
/** Gets a given coordiante in the svector. If this coordiante is null, then
    a null is returned.
    @param index The index of the entry being referenced
*/
  inline const double operator()(unsigned long index) const;

//-----------------------------------------------------------------------------
/** Gets a given coordiante in the svector. If this coordiante is null, then
    a null is returned.
    @param index The index of the entry being referenced
*/
  inline const double operator[](unsigned long index) const;

//-----------------------------------------------------------------------------
/** Gets a given coordiante in the svector. This is a slightly more
    sophisticated function than the others. When accessing an entry in a
    sparse vector, one of three things can happen: (1) v[i] = 5 is the case
    where memory for entry i must be allocated and it should be set to 5. (2)
    v[i] = 0 is the case where entry i should be erased from the vector (3)
    v[i] + 5 where no allocation or disallocation should take place. This is
    implemented by returning a 'reference' object which does all of the
    memory management.
    @param index The index of the entry being referenced.
*/
  inline reference operator()(unsigned long index);

//-----------------------------------------------------------------------------
/** Gets a given coordiante in the svector. This is a slightly more
    sophisticated function than the others. When accessing an entry in a
    sparse vector, one of three things can happen: (1) v[i] = 5 is the case
    where memory for entry i must be allocated and it should be set to 5. (2)
    v[i] = 0 is the case where entry i should be erased from the vector (3)
    v[i] + 5 where no allocation or disallocation should take place. This is
    implemented by returning a 'reference' object which does all of the
    memory management.
    @param index The index of the entry being referenced.
*/
  inline reference operator[](unsigned long index);

//*****************************************************************************
// iterators
//*****************************************************************************
  typedef std::vector<sentry>::iterator iterator;
  typedef std::vector<sentry>::const_iterator const_iterator;

//-----------------------------------------------------------------------------
/** Returns an iterator that points to the first entry in the vector.
    @return An iterator that points to the first entry in the vector.
*/
  inline iterator begin();

//-----------------------------------------------------------------------------
/** Returns a const_iterator that points to the first entry in vector. Similar
    to the non-const begin().
    @see begin().
*/
  inline const_iterator begin() const;

//-----------------------------------------------------------------------------
/** Returns an iterator that points to one past the location of the last entry
    in the vector. That is, if another iterator were to point to the last entry
    in a vector and the ++ operator would be called, then this iterator would
    equal end().
    @return An iterator that points to one position past the last reachable
    entry in the vector.
*/
  inline iterator end();

//-----------------------------------------------------------------------------
/** Returns a const_iterator that points to one position past the last entry in
    the vector. Similar to the non-const end().
    @see end().
*/
  inline const_iterator end() const;

//=============================================================================
// comparisons
//=============================================================================
//-----------------------------------------------------------------------------
/** Equality operator
    @param other The svector being compared
    @return 'true' if the two vectrors are identical
*/
  inline bool operator == (const svector& other) const;

//-----------------------------------------------------------------------------
/** Inequality operator
    @param other The svector being compared
    @return 'true' if the two vectrors are not identical
*/
  inline bool operator != (const svector& other) const;

//=============================================================================
// arithmetic operations
//=============================================================================
//-----------------------------------------------------------------------------
/** Multipilcation by a scalar. Multiplies all of the svector entries
    by some scalar.
    @param scalar The scalar to multiply all of the svector by
    @return A reference to this svector
*/
  inline svector& operator *= (const double& scalar);
  
//-----------------------------------------------------------------------------
/** Division by a scalar. Divides all of the svector entries
    by some scalar.
    @param scalar The scalar to divide all of the svector by
    @return A reference to this svector
*/
  inline svector& operator /= (const double& scalar);
  
//-----------------------------------------------------------------------------
/** Coordiante wise power. Changes every element v_i of the svector to be
    v_i^scalar.
    @param scalar The power by which to raise every svector element.
    @return A reference to this svector
*/
  inline svector& pow(const double& scalar);
  
/*============================================================================
 svector members
=============================================================================*/
 protected:
  unsigned long _size;
  std::vector<sentry> _data;
};

//*****************************************************************************
// Declaration of stream operators
//*****************************************************************************
//----------------------------------------------------------------------------
/** Output streaming operator for svector. Prints the size, then the
    number of entries, then a sequence of index-value pairs.
    @return The output stream, to allow concatinated output.
    @param os The output stream to print to.
    @param v A reference to the svector being printed.
*/
inline std::ostream& operator<< (std::ostream& os, const svector& v);

//----------------------------------------------------------------------------
/** Input streaming operator for svector.
    @return The input stream, to allow concatinated input.
    @param is The input stream to read from.
    @param s A reference to the svector being read.
*/
inline std::istream& operator>> (std::istream& is, infra::svector& v);
};

#endif
//*****************************************************************************
//                                     E O F
//*****************************************************************************
