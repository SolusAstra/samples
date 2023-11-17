#pragma once
#include "frame/ReferenceFrame.cuh"
#include "state/State.h"

class EarthMoonBarycenter
{
    Orbit* orbit;

    EarthMoonBarycenter() : state(new Keplerian) {
        
        state->a =        1.00000261;
        state.a_dot =    0.00000562;
        state.e =        0.01671123;
        state.e_dot =   -0.00004392;
        state.i =       -0.00001531;
        state.i_dot =   -0.01294668;
        state.L =    35999.37244981;
        state.L_dot =   -4.55343205;
        state.w =        0.32327364;
        state.w_dot =  -23.94362959;
        state.O =       49.55953891;
        state.O_dot =   -0.29257343;
    }


};