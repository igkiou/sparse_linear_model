//=============================================================================
// File Name: infra_ss_funcs.h
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
#ifndef _OFERD_SS_FUNCS_H_
#define _OFERD_SS_FUNCS_H_

//*****************************************************************************
// Included Files
//*****************************************************************************
#include "infra_svector.h"

namespace infra {
//-----------------------------------------------------------------------------
/** Performs svector-svector multiplication: outcome = u * v 
    @param u A constant reference to a svector
    @param v A constant reference to a svector
    @param outcome A reference to where the outcome will be stored
*/
void prod(const infra::svector& u, const infra::svector& v, double& outcome);

//-----------------------------------------------------------------------------
/** Performs svector svector multiplication and returns the answer
    @param s1 A constant reference to a sparse_svector
    @param s2 A constant reference to a sparse_svector
    @return the product
*/
double prod(const infra::svector& u, const infra::svector& v);

//-----------------------------------------------------------------------------
/** Performs svector-svector multiplication and adds the outcome to
    what is currently stored in outcome: outcome += u * v 
    @param u A constant reference to a svector
    @param v A constant reference to a svector
    @param outcome A reference to where the outcome will be stored
*/
void add_prod(const infra::svector& u, const infra::svector& v, 
              double& outcome);

//-----------------------------------------------------------------------------
/** Calculates the squared l2 distance between two svectors
    @param u A constant reference to a svector
    @param v A constant reference to a svector
    @param outcome A reference to where the outcome will be stored
*/
void dist2(const infra::svector& u, const infra::svector& v, double& outcome);

//-----------------------------------------------------------------------------
/** Calculates the squared l2 distance between two svectors
    @param u A constant reference to a svector
    @param v A constant reference to a svector
    @return The squared l2 distance between the two svectors
*/
double dist2(const infra::svector& u, const infra::svector& v);
};
#endif
//*****************************************************************************
//                                     E O F
//*****************************************************************************
